"""
Global CUDA Graph Manager for Cross-Layer Sharing with Padding Support

核心思想:
1. 所有层共享同一个 CUDA Graph Runner
2. 只录制一个最大长度的 graph，短序列自动 padding
3. 极大减少内存占用和录制开销

优势:
- 内存节省: 只有 1 个 graph vs 32层 x N个长度 = 节省 32N 倍内存
- 录制开销: 只需录制一次 vs 每层每个长度录制 = 节省 32N 倍时间
- Warmup 快速: 只需 decode 到最大长度一次
- 性能: padding 开销小，replay 速度快
"""
from __future__ import annotations

import torch
from typing import Optional, Dict

try:
    from .cudagraph_runner_padded import CudaGraphRunnerWithPadding
except ImportError:
    from cudagraph_runner_padded import CudaGraphRunnerWithPadding


class GlobalCudaGraphManager:
    """
    全局 CUDA Graph 管理器，支持跨层共享和自动 padding。

    设计:
    - 单例模式: 整个模型只有一个实例
    - 单一 runner: 只有一个 CudaGraphRunnerWithPadding
    - 所有层共享: 所有层使用同一个 runner
    - 自动 padding: 短序列自动 pad 到最大长度
    """

    _instance: Optional['GlobalCudaGraphManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device('cuda')
        # 创建共享的 CUDA stream
        self.stream = torch.cuda.Stream(self.device)

        # 单一的 runner (支持 padding)
        self.runner: Optional[CudaGraphRunnerWithPadding] = None

        # 统计信息
        self.num_warmup_calls = 0
        self.num_replay_calls = 0

        self._initialized = True

    def warmup(
        self,
        kernel_fn,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor] = None,
        k_zero: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Warmup CUDA Graph (只在第一次调用时录制)。

        Args:
            kernel_fn: attention kernel 函数
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T_max, HKV, K_packed] 量化的 K (最大长度)
            k_scale: [B, NTB_max, HKV, K] 量化 scale
            v: [B, T_max, HKV, V] Value tensor
            k_residual: [B, T_max, HKV, K] FP8 残差 (可选)
            k_zero: [B, HKV, K] 量化 zero point (可选)
            **kwargs: 其他参数

        Returns:
            output: [B, HQ, V] Attention 输出
        """
        if self.runner is None:
            # 第一次调用: 创建 runner 并录制
            self.runner = CudaGraphRunnerWithPadding(
                kernel_fn=kernel_fn,
                device=self.device,
                stream=self.stream,
            )
            self.num_warmup_calls += 1
            return self.runner.warmup(q, k_q, k_scale, v, k_residual, k_zero, **kwargs)
        elif not self.runner.is_captured:
            # Runner 已创建但未录制
            self.num_warmup_calls += 1
            return self.runner.warmup(q, k_q, k_scale, v, k_residual, k_zero, **kwargs)
        else:
            # 已经录制过，直接 replay
            self.num_replay_calls += 1
            return self.runner.replay(q, k_q, k_scale, v, k_residual, k_zero)

    def replay(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor] = None,
        k_zero: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        重放 CUDA Graph (支持自动 padding)。

        Args:
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T, HKV, K_packed] 量化的 K (T <= T_max)
            k_scale: [B, NTB, HKV, K] 量化 scale
            v: [B, T, HKV, V] Value tensor
            k_residual: [B, T, HKV, K] FP8 残差 (可选)
            k_zero: [B, HKV, K] 量化 zero point (可选)

        Returns:
            output: [B, HQ, V] Attention 输出
        """
        if self.runner is None or not self.runner.is_captured:
            raise RuntimeError(
                "CUDA Graph not captured yet. Call warmup() first."
            )

        self.num_replay_calls += 1
        return self.runner.replay(q, k_q, k_scale, v, k_residual, k_zero)

    @property
    def is_captured(self) -> bool:
        """检查 CUDA Graph 是否已录制"""
        return self.runner is not None and self.runner.is_captured

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'is_captured': self.is_captured,
            'max_seq_len': self.runner.max_seq_len if self.runner else None,
            'num_warmup_calls': self.num_warmup_calls,
            'num_replay_calls': self.num_replay_calls,
        }

    @classmethod
    def reset_instance(cls):
        """重置单例实例 (用于测试)"""
        if cls._instance is not None:
            if cls._instance.runner is not None:
                cls._instance.runner.reset()
            cls._instance._initialized = False
            cls._instance = None

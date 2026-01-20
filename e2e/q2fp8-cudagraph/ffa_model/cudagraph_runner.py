"""
CUDA Graph Runner for Q2FP8 Decode Attention

封装 CUDA Graph 的录制和重放逻辑,用于加速 decode 阶段的 attention 计算。

核心功能:
1. 录制 CUDA Graph (warmup 阶段)
2. 重放 CUDA Graph (实际推理阶段)
3. 管理输入输出 buffer
4. 处理不同序列长度的 graph 缓存
"""
from __future__ import annotations

import torch
from typing import Optional, Dict, Tuple


class CudaGraphRunner:
    """
    CUDA Graph Runner for FFA Q2FP8 Decode Attention.

    使用方法:
    1. 初始化时指定 kernel 函数和参数
    2. 调用 warmup() 录制 CUDA Graph
    3. 调用 replay() 重放 CUDA Graph
    """

    def __init__(
        self,
        kernel_fn,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        Args:
            kernel_fn: 要录制的 kernel 函数 (attn_forward_decode_quantized)
            device: CUDA 设备
            stream: CUDA stream (默认创建新的非默认 stream)
        """
        self.kernel_fn = kernel_fn
        self.device = device
        # CUDA Graph 必须在非默认 stream 上录制
        self.stream = stream or torch.cuda.Stream(device)

        # CUDA Graph 相关
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.is_captured = False

        # 输入输出 buffer (固定地址,用于 graph replay)
        self.static_q: Optional[torch.Tensor] = None
        self.static_k_q: Optional[torch.Tensor] = None
        self.static_k_scale: Optional[torch.Tensor] = None
        self.static_k_zero: Optional[torch.Tensor] = None
        self.static_v: Optional[torch.Tensor] = None
        self.static_k_residual: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None

        # 录制时的参数 (用于验证)
        self.recorded_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
        self.recorded_kwargs: Optional[Dict] = None

    def warmup(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor] = None,
        k_zero: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Warmup 阶段: 录制 CUDA Graph。

        Args:
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T, HKV, K_packed] 量化的 K
            k_scale: [B, NTB, HKV, K] 量化 scale
            v: [B, T, HKV, V] Value tensor
            k_residual: [B, T, HKV, K] FP8 残差 (可选)
            k_zero: [B, HKV, K] 量化 zero point (可选,对称量化不需要)
            **kwargs: 其他参数 (scale, BS, SBS, delta, etc.)

        Returns:
            output: [B, HQ, V] Attention 输出
        """
        if self.is_captured:
            raise RuntimeError("CUDA Graph already captured. Use replay() instead.")

        # 记录输入形状和参数
        self.recorded_shapes = {
            'q': q.shape,
            'k_q': k_q.shape,
            'k_scale': k_scale.shape,
            'v': v.shape,
            'k_residual': k_residual.shape if k_residual is not None else None,
            'k_zero': k_zero.shape if k_zero is not None else None,
        }
        self.recorded_kwargs = kwargs.copy()

        # 创建静态 buffer (固定地址)
        self.static_q = torch.zeros_like(q)
        self.static_k_q = torch.zeros_like(k_q)
        self.static_k_scale = torch.zeros_like(k_scale)
        self.static_v = torch.zeros_like(v)

        if k_residual is not None:
            self.static_k_residual = torch.zeros_like(k_residual)
        else:
            self.static_k_residual = None

        if k_zero is not None:
            self.static_k_zero = torch.zeros_like(k_zero)
        else:
            self.static_k_zero = None

        # 预分配输出 buffer
        B, _, HQ, K = q.shape
        _, _, _, V = v.shape
        self.static_output = torch.zeros((B, HQ, V), dtype=q.dtype, device=self.device)

        # 复制输入到静态 buffer
        self.static_q.copy_(q)
        self.static_k_q.copy_(k_q)
        self.static_k_scale.copy_(k_scale)
        self.static_v.copy_(v)
        if k_residual is not None:
            self.static_k_residual.copy_(k_residual)
        if k_zero is not None:
            self.static_k_zero.copy_(k_zero)

        # 录制 CUDA Graph
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph, stream=self.stream):
            # 调用 kernel 函数
            output = self.kernel_fn(
                q=self.static_q,
                k_q=self.static_k_q,
                k_scale=self.static_k_scale,
                v=self.static_v,
                k_residual=self.static_k_residual,
                k_zero=self.static_k_zero,
                **kwargs,
            )
            self.static_output.copy_(output)

        self.is_captured = True

        # 返回输出
        return self.static_output.clone()

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
        重放 CUDA Graph。

        Args:
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T, HKV, K_packed] 量化的 K
            k_scale: [B, NTB, HKV, K] 量化 scale
            v: [B, T, HKV, V] Value tensor
            k_residual: [B, T, HKV, K] FP8 残差 (可选)
            k_zero: [B, HKV, K] 量化 zero point (可选)

        Returns:
            output: [B, HQ, V] Attention 输出
        """
        if not self.is_captured:
            raise RuntimeError("CUDA Graph not captured yet. Call warmup() first.")

        # 验证输入形状
        if q.shape != self.recorded_shapes['q']:
            raise ValueError(f"q shape mismatch: expected {self.recorded_shapes['q']}, got {q.shape}")
        if k_q.shape != self.recorded_shapes['k_q']:
            raise ValueError(f"k_q shape mismatch: expected {self.recorded_shapes['k_q']}, got {k_q.shape}")
        if k_scale.shape != self.recorded_shapes['k_scale']:
            raise ValueError(f"k_scale shape mismatch: expected {self.recorded_shapes['k_scale']}, got {k_scale.shape}")
        if v.shape != self.recorded_shapes['v']:
            raise ValueError(f"v shape mismatch: expected {self.recorded_shapes['v']}, got {v.shape}")

        # 复制输入到静态 buffer
        self.static_q.copy_(q)
        self.static_k_q.copy_(k_q)
        self.static_k_scale.copy_(k_scale)
        self.static_v.copy_(v)
        if k_residual is not None and self.static_k_residual is not None:
            self.static_k_residual.copy_(k_residual)
        if k_zero is not None and self.static_k_zero is not None:
            self.static_k_zero.copy_(k_zero)

        # 重放 graph
        self.graph.replay()

        # 返回输出
        return self.static_output.clone()

    def reset(self):
        """重置 CUDA Graph (释放资源)"""
        self.graph = None
        self.is_captured = False
        self.static_q = None
        self.static_k_q = None
        self.static_k_scale = None
        self.static_k_zero = None
        self.static_v = None
        self.static_k_residual = None
        self.static_output = None
        self.recorded_shapes = None
        self.recorded_kwargs = None


class MultiLengthCudaGraphRunner:
    """
    支持多个序列长度的 CUDA Graph Runner。

    为不同的序列长度录制不同的 CUDA Graph,自动选择合适的 graph 进行重放。
    """

    def __init__(
        self,
        kernel_fn,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.kernel_fn = kernel_fn
        self.device = device
        # CUDA Graph 必须在非默认 stream 上录制
        self.stream = stream or torch.cuda.Stream(device)

        # 存储不同长度的 runner
        self.runners: Dict[int, CudaGraphRunner] = {}

    def get_or_create_runner(self, seq_len: int) -> CudaGraphRunner:
        """获取或创建指定序列长度的 runner"""
        if seq_len not in self.runners:
            self.runners[seq_len] = CudaGraphRunner(
                kernel_fn=self.kernel_fn,
                device=self.device,
                stream=self.stream,
            )
        return self.runners[seq_len]

    def warmup(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor] = None,
        k_zero: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Warmup 指定序列长度的 CUDA Graph"""
        seq_len = k_q.shape[1]  # T dimension
        runner = self.get_or_create_runner(seq_len)
        return runner.warmup(q, k_q, k_scale, v, k_residual, k_zero, **kwargs)

    def replay(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor] = None,
        k_zero: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """重放指定序列长度的 CUDA Graph"""
        seq_len = k_q.shape[1]  # T dimension

        if seq_len not in self.runners:
            raise RuntimeError(
                f"No CUDA Graph captured for seq_len={seq_len}. "
                f"Available lengths: {list(self.runners.keys())}"
            )

        runner = self.runners[seq_len]
        return runner.replay(q, k_q, k_scale, v, k_residual, k_zero)

    def is_captured(self, seq_len: int) -> bool:
        """检查指定序列长度的 graph 是否已录制"""
        return seq_len in self.runners and self.runners[seq_len].is_captured

    def reset(self):
        """重置所有 CUDA Graph"""
        for runner in self.runners.values():
            runner.reset()
        self.runners.clear()

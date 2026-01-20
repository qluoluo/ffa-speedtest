"""
CUDA Graph Runner with Padding Support for Q2FP8 Decode Attention

支持自动 padding 的 CUDA Graph Runner，只需要 capture 一次最大长度的 graph。

核心功能:
1. 录制最大长度的 CUDA Graph (warmup 阶段)
2. 自动 padding 短序列到最大长度
3. 重放 CUDA Graph (实际推理阶段)
4. 管理输入输出 buffer
"""
from __future__ import annotations

import torch
from typing import Optional, Dict, Tuple


class CudaGraphRunnerWithPadding:
    """
    CUDA Graph Runner with automatic padding support.

    使用方法:
    1. 初始化时指定 kernel 函数和参数
    2. 调用 warmup() 录制最大长度的 CUDA Graph
    3. 调用 replay() 重放 CUDA Graph (自动 padding 短序列)
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

        # 录制时的参数 (用于验证和 padding)
        self.recorded_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
        self.recorded_kwargs: Optional[Dict] = None
        self.max_seq_len: Optional[int] = None  # 录制时的最大序列长度

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
        Warmup 阶段: 录制 CUDA Graph (使用最大序列长度)。

        Args:
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T_max, HKV, K_packed] 量化的 K (最大长度)
            k_scale: [B, NTB_max, HKV, K] 量化 scale (最大长度)
            v: [B, T_max, HKV, V] Value tensor (最大长度)
            k_residual: [B, T_max, HKV, K] FP8 残差 (可选)
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
        self.max_seq_len = k_q.shape[1]  # T dimension

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
        if not self.is_captured:
            raise RuntimeError("CUDA Graph not captured yet. Call warmup() first.")

        # 验证 q 形状 (不需要 padding)
        if q.shape != self.recorded_shapes['q']:
            raise ValueError(f"q shape mismatch: expected {self.recorded_shapes['q']}, got {q.shape}")

        # 获取当前序列长度
        current_seq_len = k_q.shape[1]

        # 检查是否需要 padding
        if current_seq_len > self.max_seq_len:
            raise ValueError(
                f"Current seq_len ({current_seq_len}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please warmup with a longer sequence."
            )

        # Padding 到最大长度
        if current_seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - current_seq_len

            # Pad k_q: [B, T, HKV, K_packed] -> [B, T_max, HKV, K_packed]
            k_q_padded = torch.nn.functional.pad(k_q, (0, 0, 0, 0, 0, pad_len), value=0)

            # Pad k_scale: [B, NTB, HKV, K] -> [B, NTB_max, HKV, K]
            # 注意: k_scale 的第二维是 block 数量，需要计算
            BS = self.recorded_kwargs.get('BS', 128)
            current_num_blocks = (current_seq_len + BS - 1) // BS
            max_num_blocks = (self.max_seq_len + BS - 1) // BS
            pad_blocks = max_num_blocks - current_num_blocks
            k_scale_padded = torch.nn.functional.pad(k_scale, (0, 0, 0, 0, 0, pad_blocks), value=1.0)  # scale 用 1.0 填充

            # Pad v: [B, T, HKV, V] -> [B, T_max, HKV, V]
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len), value=0)

            # Pad k_residual if exists
            if k_residual is not None:
                k_residual_padded = torch.nn.functional.pad(k_residual, (0, 0, 0, 0, 0, pad_len), value=0)
            else:
                k_residual_padded = None
        else:
            # 不需要 padding
            k_q_padded = k_q
            k_scale_padded = k_scale
            v_padded = v
            k_residual_padded = k_residual

        # 复制输入到静态 buffer
        self.static_q.copy_(q)
        self.static_k_q.copy_(k_q_padded)
        self.static_k_scale.copy_(k_scale_padded)
        self.static_v.copy_(v_padded)
        if k_residual_padded is not None and self.static_k_residual is not None:
            self.static_k_residual.copy_(k_residual_padded)
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
        self.max_seq_len = None

"""
Sparse Attention with FlashInfer integration.

主模块：结合 Triton 采样筛选和 FlashInfer 精确计算。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .kernels import (
    sample_k_fp16,
    attn_forward_decode_sample4,
    SAMPLE_OFFSETS,
    NUM_SAMPLES,
)
from .utils import FlashInferSparseDecodeWrapper, PagedKVCache, continuous_to_paged


class SparseAttentionWithFlashInfer(nn.Module):
    """
    结合 Triton 采样筛选和 FlashInfer 精确计算的稀疏注意力模块.

    架构设计类似于 Quest:
    1. 使用 Triton kernel 快速筛选重要的 KV block
    2. (可选) 使用 FlashInfer 对选中的 block 进行精确计算

    两种工作模式:
    1. Triton-only: 筛选 + Triton 精确计算 (默认)
    2. Hybrid: Triton 筛选 + FlashInfer 精确计算

    Example:
        >>> sparse_attn = SparseAttentionWithFlashInfer(
        ...     num_heads=32,
        ...     head_dim=128,
        ...     page_size=128,
        ... )
        >>> output = sparse_attn(q, k, v, delta=5.0)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int = 128,
        num_kv_heads: Optional[int] = None,
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.float16,
        use_flashinfer: bool = False,
    ):
        """
        初始化 SparseAttentionWithFlashInfer.

        Args:
            num_heads: Query 头数
            head_dim: 头维度
            page_size: KV 页面大小 (与 Triton kernel 的 BS 对应)
            num_kv_heads: KV 头数 (用于 GQA)，默认等于 num_heads
            device: 设备
            dtype: 数据类型
            use_flashinfer: 是否使用 FlashInfer 进行精确计算
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.use_flashinfer = use_flashinfer

        # 计算 group 数 (用于 GQA)
        assert num_heads % self.num_kv_heads == 0
        self.num_groups = num_heads // self.num_kv_heads

        # 如果启用 FlashInfer，初始化 wrapper
        self._flashinfer_wrapper: Optional[FlashInferSparseDecodeWrapper] = None

    def _init_flashinfer(self, max_pages: int) -> None:
        """懒初始化 FlashInfer wrapper."""
        if self._flashinfer_wrapper is None and self.use_flashinfer:
            self._flashinfer_wrapper = FlashInferSparseDecodeWrapper(
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                max_pages=max_pages,
                device=self.device,
                dtype=self.dtype,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        delta: float = 5.0,
        scale: Optional[float] = None,
        return_skip_ratio: bool = False,
        max_kept_ratio: float = 0.2,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        执行稀疏注意力计算.

        Args:
            q: Query, shape: [B, 1, HQ, K] (decode 模式)
            k: Key, shape: [B, T, HKV, K]
            v: Value, shape: [B, T, HKV, K]
            delta: 阈值 delta 参数
            scale: softmax scale，默认 1/sqrt(head_dim)
            return_skip_ratio: 是否返回跳过比例
            max_kept_ratio: 最大保留 block 比例
            **kwargs: 传递给底层 kernel 的其他参数

        Returns:
            output: 注意力输出, shape: [B, HQ, V]
            skip_ratio (optional): 跳过的 block 比例
        """
        B, Tq, HQ, K = q.shape
        _, T, HKV, _ = k.shape
        V = v.shape[-1]

        assert Tq == 1, "Currently only decode mode (Tq=1) is supported"
        assert HQ == self.num_heads
        assert HKV == self.num_kv_heads

        if scale is None:
            scale = 1.0 / math.sqrt(K)

        # 1. 提取采样 K (用于快速筛选)
        k_sample = sample_k_fp16(k, BS=self.page_size, sample_offsets=SAMPLE_OFFSETS)
        # k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K]

        # 2. 创建 dummy scale (兼容接口)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros(
            (B, num_blocks, HKV, K), device=k.device, dtype=k.dtype
        )

        # 3. 调用 Triton kernel 进行筛选和计算
        result = attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            scale=scale,
            BS=self.page_size,
            delta=delta,
            return_skip_ratio=return_skip_ratio,
            max_kept_ratio=max_kept_ratio,
            **kwargs,
        )

        if return_skip_ratio:
            output, skip_ratio = result
            return output, skip_ratio
        else:
            return result

    def forward_with_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        delta: float = 5.0,
        scale: Optional[float] = None,
        max_kept_ratio: float = 0.2,
        pos_encoding_mode: str = "NONE",
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        使用 FlashInfer 进行精确计算的混合模式.

        流程:
        1. Triton kernel 筛选出重要的 block
        2. 将 KV 转换为 paged 格式
        3. FlashInfer 仅对选中的 block 计算

        Args:
            q: Query, shape: [B, 1, HQ, K]
            k: Key, shape: [B, T, HKV, K]
            v: Value, shape: [B, T, HKV, K]
            delta: 阈值参数
            scale: softmax scale
            max_kept_ratio: 最大保留比例
            pos_encoding_mode: 位置编码模式
            rope_scale: RoPE scale
            rope_theta: RoPE theta

        Returns:
            output: 注意力输出
        """
        if not self.use_flashinfer:
            raise RuntimeError(
                "FlashInfer mode is not enabled. "
                "Set use_flashinfer=True in constructor."
            )

        B, Tq, HQ, K = q.shape
        _, T, HKV, _ = k.shape
        V = v.shape[-1]

        assert Tq == 1, "Only decode mode is supported"

        if scale is None:
            scale = 1.0 / math.sqrt(K)

        # 1. 提取采样 K
        k_sample = sample_k_fp16(k, BS=self.page_size)
        num_blocks = k_sample.shape[1]

        # 2. 使用 Triton 计算阈值和筛选 (简化版)
        # 这里我们使用完整的 Triton kernel，它会返回 block_mask
        k_sample_scale = torch.zeros(
            (B, num_blocks, HKV, K), device=k.device, dtype=k.dtype
        )

        # 调用 Triton kernel 获取筛选结果
        output, skip_ratio = attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            scale=scale,
            BS=self.page_size,
            delta=delta,
            return_skip_ratio=True,
            max_kept_ratio=max_kept_ratio,
        )

        # 目前直接返回 Triton 的结果
        # TODO: 实现真正的 Triton 筛选 + FlashInfer 精确计算的混合模式
        return output


class SparseAttentionConfig:
    """稀疏注意力配置类."""

    def __init__(
        self,
        page_size: int = 128,
        delta: float = 5.0,
        max_kept_ratio: float = 0.2,
        use_flashinfer: bool = False,
        sample_offsets: Optional[list] = None,
    ):
        """
        Args:
            page_size: KV 页面大小
            delta: 阈值参数
            max_kept_ratio: 最大保留 block 比例
            use_flashinfer: 是否使用 FlashInfer
            sample_offsets: 采样点偏移，默认 [0, 32, 64, 96]
        """
        self.page_size = page_size
        self.delta = delta
        self.max_kept_ratio = max_kept_ratio
        self.use_flashinfer = use_flashinfer
        self.sample_offsets = sample_offsets or SAMPLE_OFFSETS

    def to_dict(self) -> dict:
        return {
            "page_size": self.page_size,
            "delta": self.delta,
            "max_kept_ratio": self.max_kept_ratio,
            "use_flashinfer": self.use_flashinfer,
            "sample_offsets": self.sample_offsets,
        }


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: Optional[SparseAttentionConfig] = None,
    **kwargs,
) -> torch.Tensor:
    """
    函数式接口的稀疏注意力.

    Args:
        q: Query, [B, 1, HQ, K]
        k: Key, [B, T, HKV, K]
        v: Value, [B, T, HKV, K]
        config: 配置对象
        **kwargs: 覆盖配置的参数

    Returns:
        output: [B, HQ, V]
    """
    if config is None:
        config = SparseAttentionConfig()

    # 合并配置和 kwargs
    params = config.to_dict()
    params.update(kwargs)

    # 提取采样 K
    k_sample = sample_k_fp16(
        k, BS=params["page_size"], sample_offsets=params["sample_offsets"]
    )

    B, num_blocks, HKV, _, K = k_sample.shape
    k_sample_scale = torch.zeros(
        (B, num_blocks, HKV, K), device=k.device, dtype=k.dtype
    )

    # 调用 Triton kernel
    output = attn_forward_decode_sample4(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale,
        k_full=k,
        v=v,
        BS=params["page_size"],
        delta=params["delta"],
        max_kept_ratio=params["max_kept_ratio"],
    )

    return output

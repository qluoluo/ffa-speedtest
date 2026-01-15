"""
FlashInfer wrapper for precise attention computation on selected KV blocks.

This module provides a Python wrapper around FlashInfer's BatchDecodeWithPagedKVCacheWrapper,
enabling integration with Triton-based sparse attention algorithms.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

try:
    import flashinfer
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    BatchDecodeWithPagedKVCacheWrapper = None


class FlashInferDecodeWrapper:
    """
    封装 FlashInfer 的 BatchDecodeWithPagedKVCacheWrapper，
    用于对选中的 KV 页面执行精确的注意力计算。

    类似于 Quest 的 BatchDecodeWithPagedKVCachePyTorchWrapper，
    但使用 FlashInfer 的 Python API 而不是 C++ 绑定。

    使用流程:
        1. 初始化: wrapper = FlashInferDecodeWrapper(...)
        2. 规划: wrapper.plan(indptr, indices, last_page_len, ...)
        3. 执行: output = wrapper.run(q, kv_cache)
        4. 结束: wrapper.end_forward()

    Example:
        >>> wrapper = FlashInferDecodeWrapper(
        ...     num_qo_heads=32,
        ...     num_kv_heads=8,
        ...     head_dim=128,
        ...     page_size=16,
        ...     device="cuda:0"
        ... )
        >>> # 假设已经准备好 paged kv cache 和 indices
        >>> wrapper.plan(indptr, indices, last_page_len, ...)
        >>> output = wrapper.run(q, kv_cache)
    """

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        device: Union[str, torch.device] = "cuda:0",
        kv_layout: str = "NHD",
        workspace_size_mb: int = 128,
        dtype: torch.dtype = torch.float16,
    ):
        """
        初始化 FlashInfer Decode Wrapper.

        Args:
            num_qo_heads: Query/Output 头数
            num_kv_heads: Key/Value 头数 (用于 GQA)
            head_dim: 每个头的维度
            page_size: KV cache 的页面大小
            device: 设备
            kv_layout: KV 布局, "NHD" 或 "HND"
            workspace_size_mb: 工作空间大小 (MB)
            dtype: 数据类型
        """
        if not FLASHINFER_AVAILABLE:
            raise ImportError(
                "FlashInfer is not installed. "
                "Please install it with: pip install flashinfer"
            )

        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.kv_layout = kv_layout
        self.dtype = dtype

        # 创建工作空间 buffer
        self._workspace_buffer = torch.zeros(
            workspace_size_mb * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device
        )

        # 创建 FlashInfer wrapper
        self._wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=self._workspace_buffer,
            kv_layout=kv_layout,
        )

        self._planned = False

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> None:
        """
        规划 batch decode attention.

        这个方法会创建辅助数据结构，可以在多个层之间复用。

        Args:
            indptr: 页面索引的 indptr, shape: [batch_size + 1]
            indices: 页面索引, shape: [num_pages]
            last_page_len: 每个请求最后一页的有效长度, shape: [batch_size]
            pos_encoding_mode: 位置编码模式 ("NONE", "ROPE_LLAMA", "ALIBI")
            sm_scale: softmax scale, 默认为 1/sqrt(head_dim)
            rope_scale: RoPE scale
            rope_theta: RoPE theta
        """
        self._wrapper.plan(
            indptr=indptr.to(torch.int32),
            indices=indices.to(torch.int32),
            last_page_len=last_page_len.to(torch.int32),
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        self._planned = True

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行 batch decode attention.

        Args:
            q: Query tensor, shape: [batch_size, num_qo_heads, head_dim]
            paged_kv_cache: Paged KV cache
                - 如果 kv_layout="NHD": shape [max_num_pages, 2, page_size, num_kv_heads, head_dim]
                - 如果 kv_layout="HND": shape [max_num_pages, 2, num_kv_heads, page_size, head_dim]
            return_lse: 是否返回 log-sum-exp

        Returns:
            output: 注意力输出, shape: [batch_size, num_qo_heads, head_dim]
            lse (optional): log-sum-exp, shape: [batch_size, num_qo_heads]
        """
        if not self._planned:
            raise RuntimeError("Must call plan() before run()")

        return self._wrapper.run(q, paged_kv_cache, return_lse=return_lse)

    def end_forward(self) -> None:
        """结束当前的 forward pass，释放资源."""
        self._planned = False


class FlashInferSparseDecodeWrapper:
    """
    专门用于稀疏注意力的 FlashInfer wrapper.

    这个类封装了 FlashInfer，使其可以只计算选中的 KV 页面，
    类似于 Quest 中的 decode_sparse_attn 功能。

    核心思想:
        1. 接收 top-k 选中的页面索引
        2. 构建只包含这些页面的 indptr/indices
        3. 调用 FlashInfer 进行精确计算

    Example:
        >>> wrapper = FlashInferSparseDecodeWrapper(...)
        >>> # kept_indices 是通过 Triton 筛选得到的 top-k 页面索引
        >>> output = wrapper.forward(q, kv_cache, kept_indices, kept_counts)
    """

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        device: Union[str, torch.device] = "cuda:0",
        kv_layout: str = "NHD",
        dtype: torch.dtype = torch.float16,
    ):
        """
        初始化 sparse decode wrapper.

        Args:
            num_qo_heads: Query/Output 头数
            num_kv_heads: Key/Value 头数
            head_dim: 头维度
            page_size: 页面大小
            max_pages: 最大页面数 (用于预分配 buffer)
            device: 设备
            kv_layout: KV 布局
            dtype: 数据类型
        """
        if not FLASHINFER_AVAILABLE:
            raise ImportError("FlashInfer is not installed.")

        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = torch.device(device) if isinstance(device, str) else device
        self.kv_layout = kv_layout
        self.dtype = dtype

        # 工作空间
        self._workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # FlashInfer wrapper
        self._wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=self._workspace_buffer,
            kv_layout=kv_layout,
        )

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kept_indices: torch.Tensor,
        kept_counts: torch.Tensor,
        last_page_len: Optional[torch.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        对选中的页面执行精确注意力计算.

        Args:
            q: Query, shape: [batch_size, num_qo_heads, head_dim]
            paged_kv_cache: 完整的 paged KV cache
            kept_indices: 保留的页面索引, shape: [batch_size, num_kv_heads, max_kept]
                         或 [batch_size, max_kept] (如果所有头共享)
            kept_counts: 每个请求保留的页面数, shape: [batch_size] 或 [batch_size, num_kv_heads]
            last_page_len: 最后一页的有效长度, shape: [batch_size]
            pos_encoding_mode: 位置编码模式
            sm_scale: softmax scale
            rope_scale: RoPE scale
            rope_theta: RoPE theta

        Returns:
            output: 注意力输出, shape: [batch_size, num_qo_heads, head_dim]
        """
        batch_size = q.shape[0]

        # 处理 kept_indices 的维度
        if kept_indices.dim() == 2:
            # [batch_size, max_kept] -> 所有头共享
            kept_indices = kept_indices.unsqueeze(1).expand(-1, self.num_kv_heads, -1)

        if kept_counts.dim() == 1:
            kept_counts = kept_counts.unsqueeze(1).expand(-1, self.num_kv_heads)

        # 构建 indptr
        # 对于 batch_size=1 的情况
        if batch_size == 1:
            num_kept = kept_counts[0, 0].item()
            indices = kept_indices[0, 0, :num_kept].contiguous()

            indptr = torch.tensor([0, num_kept], dtype=torch.int32, device=self.device)

            if last_page_len is None:
                last_page_len = torch.tensor([self.page_size], dtype=torch.int32, device=self.device)

            # 规划和执行
            self._wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                pos_encoding_mode=pos_encoding_mode,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                sm_scale=sm_scale,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
            )

            output = self._wrapper.run(q, paged_kv_cache)
            return output

        else:
            # 批量处理 - 需要合并所有请求的索引
            # 构建 indptr: [0, count_0, count_0+count_1, ...]
            counts = kept_counts[:, 0]  # 假设所有头使用相同的计数
            indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            indptr[1:] = torch.cumsum(counts, dim=0)

            total_pages = indptr[-1].item()
            indices = torch.zeros(total_pages, dtype=torch.int32, device=self.device)

            # 填充 indices
            offset = 0
            for b in range(batch_size):
                num_kept = counts[b].item()
                indices[offset:offset + num_kept] = kept_indices[b, 0, :num_kept]
                offset += num_kept

            if last_page_len is None:
                last_page_len = torch.full(
                    (batch_size,), self.page_size, dtype=torch.int32, device=self.device
                )

            self._wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                pos_encoding_mode=pos_encoding_mode,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                sm_scale=sm_scale,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
            )

            output = self._wrapper.run(q, paged_kv_cache)
            return output


def create_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    kv_layout: str = "NHD",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将连续的 KV tensor 转换为 paged KV cache 格式.

    Args:
        k: Key tensor, shape: [batch_size, seq_len, num_kv_heads, head_dim]
        v: Value tensor, shape: [batch_size, seq_len, num_kv_heads, head_dim]
        page_size: 页面大小
        kv_layout: "NHD" 或 "HND"

    Returns:
        paged_kv_cache: Paged KV cache tensor
        indices: 页面索引
        indptr: 页面 indptr
    """
    B, T, H, D = k.shape
    num_pages = (T + page_size - 1) // page_size

    # Pad to multiple of page_size
    pad_len = num_pages * page_size - T
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))

    # Reshape to pages
    # k, v: [B, num_pages, page_size, H, D]
    k_pages = k.view(B, num_pages, page_size, H, D)
    v_pages = v.view(B, num_pages, page_size, H, D)

    if kv_layout == "NHD":
        # [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        # 对于 batch_size=1，直接使用 num_pages 作为 max_num_pages
        paged_kv = torch.stack([k_pages[0], v_pages[0]], dim=1)  # [num_pages, 2, page_size, H, D]
    else:
        # [max_num_pages, 2, num_kv_heads, page_size, head_dim]
        k_pages = k_pages.transpose(2, 3)  # [B, num_pages, H, page_size, D]
        v_pages = v_pages.transpose(2, 3)
        paged_kv = torch.stack([k_pages[0], v_pages[0]], dim=1)

    # 对于单个请求，indices 就是 0, 1, 2, ..., num_pages-1
    indices = torch.arange(num_pages, dtype=torch.int32, device=k.device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=k.device)
    last_page_len = torch.tensor([T % page_size or page_size], dtype=torch.int32, device=k.device)

    return paged_kv, indices, indptr, last_page_len

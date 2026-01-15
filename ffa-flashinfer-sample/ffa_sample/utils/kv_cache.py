"""
Paged KV Cache management utilities.

提供与 FlashInfer 兼容的分页 KV 缓存管理。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


class PagedKVCache:
    """
    分页 KV 缓存管理器.

    实现类似于 Quest 中 KvCache 的功能，但更加轻量化，
    专注于与 FlashInfer 和 Triton kernel 的集成。

    数据布局:
        - NHD: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        - HND: [max_num_pages, 2, num_kv_heads, page_size, head_dim]

    Example:
        >>> cache = PagedKVCache(
        ...     max_num_pages=1024,
        ...     page_size=16,
        ...     num_kv_heads=8,
        ...     head_dim=128,
        ...     device="cuda:0"
        ... )
        >>> # 分配页面
        >>> page_indices = cache.allocate_pages(num_pages=10)
        >>> # 写入 KV
        >>> cache.append(k, v, page_indices, offset=0)
    """

    def __init__(
        self,
        max_num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.float16,
        kv_layout: str = "NHD",
    ):
        """
        初始化 PagedKVCache.

        Args:
            max_num_pages: 最大页面数
            page_size: 每页的 token 数
            num_kv_heads: KV 头数
            head_dim: 头维度
            device: 设备
            dtype: 数据类型
            kv_layout: "NHD" 或 "HND"
        """
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.kv_layout = kv_layout

        # 分配 KV 缓存存储
        if kv_layout == "NHD":
            # [max_num_pages, 2, page_size, num_kv_heads, head_dim]
            self._data = torch.zeros(
                (max_num_pages, 2, page_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=self.device,
            )
        else:
            # [max_num_pages, 2, num_kv_heads, page_size, head_dim]
            self._data = torch.zeros(
                (max_num_pages, 2, num_kv_heads, page_size, head_dim),
                dtype=dtype,
                device=self.device,
            )

        # 页面分配状态
        self._allocated = torch.zeros(max_num_pages, dtype=torch.bool, device=self.device)
        self._next_free = 0

    @property
    def data(self) -> torch.Tensor:
        """返回底层的 KV 缓存数据."""
        return self._data

    def allocate_pages(self, num_pages: int) -> torch.Tensor:
        """
        分配指定数量的页面.

        Args:
            num_pages: 需要分配的页面数

        Returns:
            page_indices: 分配的页面索引, shape: [num_pages]
        """
        if self._next_free + num_pages > self.max_num_pages:
            raise RuntimeError(
                f"Cannot allocate {num_pages} pages. "
                f"Only {self.max_num_pages - self._next_free} pages available."
            )

        indices = torch.arange(
            self._next_free,
            self._next_free + num_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self._allocated[indices] = True
        self._next_free += num_pages

        return indices

    def free_pages(self, page_indices: torch.Tensor) -> None:
        """
        释放指定的页面.

        注意: 当前实现是简单的标记，不支持碎片化回收。

        Args:
            page_indices: 要释放的页面索引
        """
        self._allocated[page_indices] = False

    def reset(self) -> None:
        """重置缓存，释放所有页面."""
        self._allocated.fill_(False)
        self._next_free = 0
        self._data.zero_()

    def append_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        page_indices: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> None:
        """
        将 KV 追加到指定的页面.

        Args:
            k: Key tensor, shape: [seq_len, num_kv_heads, head_dim]
            v: Value tensor, shape: [seq_len, num_kv_heads, head_dim]
            page_indices: 目标页面索引, shape: [num_pages]
            positions: 每个 token 在页面内的位置, shape: [seq_len]
                      如果为 None，假设顺序填充
        """
        seq_len = k.shape[0]
        num_pages = len(page_indices)

        if positions is None:
            # 假设顺序填充
            for i, page_idx in enumerate(page_indices):
                start = i * self.page_size
                end = min(start + self.page_size, seq_len)
                length = end - start

                if length > 0:
                    if self.kv_layout == "NHD":
                        self._data[page_idx, 0, :length] = k[start:end]
                        self._data[page_idx, 1, :length] = v[start:end]
                    else:
                        self._data[page_idx, 0, :, :length] = k[start:end].transpose(0, 1)
                        self._data[page_idx, 1, :, :length] = v[start:end].transpose(0, 1)
        else:
            # 使用指定位置
            for token_idx in range(seq_len):
                page_num = token_idx // self.page_size
                pos_in_page = token_idx % self.page_size

                if page_num < num_pages:
                    page_idx = page_indices[page_num]
                    if self.kv_layout == "NHD":
                        self._data[page_idx, 0, pos_in_page] = k[token_idx]
                        self._data[page_idx, 1, pos_in_page] = v[token_idx]
                    else:
                        self._data[page_idx, 0, :, pos_in_page] = k[token_idx]
                        self._data[page_idx, 1, :, pos_in_page] = v[token_idx]

    def get_page(self, page_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定页面的 K 和 V.

        Args:
            page_idx: 页面索引

        Returns:
            k: Key, shape 取决于 kv_layout
            v: Value, shape 取决于 kv_layout
        """
        return self._data[page_idx, 0], self._data[page_idx, 1]

    @property
    def num_allocated_pages(self) -> int:
        """返回已分配的页面数."""
        return self._allocated.sum().item()

    @property
    def num_free_pages(self) -> int:
        """返回可用的页面数."""
        return self.max_num_pages - self._next_free


def continuous_to_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    kv_layout: str = "NHD",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将连续的 KV tensor 转换为分页格式.

    Args:
        k: Key, shape: [batch_size, seq_len, num_kv_heads, head_dim]
        v: Value, shape: [batch_size, seq_len, num_kv_heads, head_dim]
        page_size: 页面大小
        kv_layout: "NHD" 或 "HND"

    Returns:
        paged_kv: 分页 KV cache
        indices: 页面索引
        indptr: 页面 indptr
        last_page_len: 最后一页的有效长度
    """
    B, T, H, D = k.shape
    num_pages = (T + page_size - 1) // page_size
    last_page_len_val = T % page_size or page_size

    # Pad to multiple of page_size
    pad_len = num_pages * page_size - T
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))

    # Reshape: [B, num_pages, page_size, H, D]
    k_paged = k.view(B, num_pages, page_size, H, D)
    v_paged = v.view(B, num_pages, page_size, H, D)

    # 目前只支持 batch_size=1
    if B != 1:
        raise NotImplementedError("Batch size > 1 not yet supported")

    if kv_layout == "NHD":
        # [num_pages, 2, page_size, H, D]
        paged_kv = torch.stack([k_paged[0], v_paged[0]], dim=1)
    else:
        # [num_pages, 2, H, page_size, D]
        k_paged = k_paged.transpose(2, 3)
        v_paged = v_paged.transpose(2, 3)
        paged_kv = torch.stack([k_paged[0], v_paged[0]], dim=1)

    indices = torch.arange(num_pages, dtype=torch.int32, device=k.device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=k.device)
    last_page_len = torch.tensor([last_page_len_val], dtype=torch.int32, device=k.device)

    return paged_kv, indices, indptr, last_page_len

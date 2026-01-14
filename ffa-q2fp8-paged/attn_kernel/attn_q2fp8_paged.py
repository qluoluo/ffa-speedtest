from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym"


@dataclass
class PagedKVCache:
    k_q: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    page_lens: torch.Tensor
    k_page_absmax: torch.Tensor
    k_page_sumabs: torch.Tensor
    k_page_l2: torch.Tensor
    block_table: torch.Tensor
    page_counts: torch.Tensor
    page_size: int


@triton.jit
def compute_page_meta_q2_packed(
    k_q,
    k_scale,
    page_ids,
    page_lens,
    k_page_absmax,
    k_page_sumabs,
    k_page_l2,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_kp,
    stride_scale_h,
    stride_scale_k,
    stride_meta_b,
    stride_meta_h,
    stride_pid,
    stride_plen,
    Npage,
    HKV: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    SBS: tl.constexpr,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    BLOCK_T: tl.constexpr = 16,
):
    pid_page = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    page_id = tl.load(page_ids + pid_page * stride_pid).to(tl.int32)
    page_mask = (pid_page < Npage) & (page_id >= 0) & (page_id < Npage)

    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    absmax = tl.zeros((), tl.float32)
    sumabs = tl.zeros((), tl.float32)
    sumsq = tl.zeros((), tl.float32)

    scale_base = k_scale + pid_hkv * stride_scale_h

    valid_len = tl.load(page_lens + page_id * stride_plen, mask=page_mask, other=0)
    for t_block in tl.static_range(0, SBS, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        t_mask = page_mask & (offs_t < valid_len)

        base_k = (
            page_id * stride_k_b
            + pid_hkv * stride_k_h
            + offs_t * stride_k_t
        )

        offs_k_base = tl.arange(0, BK)
        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shift = (offs_k % VALS_PER_BYTE) * K_BITS

            k_ptrs = k_q + base_k[None, :] + pack_idx[:, None] * stride_kp
            kq = tl.load(
                k_ptrs,
                mask=k_mask[:, None] & t_mask[None, :],
                other=0,
            ).to(tl.int32)
            kq = ((kq >> pack_shift[:, None]) & QMAX).to(tl.float32)
            kq = kq - QZERO

            scale = tl.load(scale_base + offs_k * stride_scale_k, mask=k_mask, other=0.0)
            k_scaled = kq * scale[:, None]

            abs_k = tl.abs(k_scaled)
            block_max = tl.max(abs_k, axis=0)
            absmax = tl.maximum(absmax, tl.max(block_max, axis=0))
            sumabs += tl.sum(tl.sum(abs_k, axis=0), axis=0)
            sumsq += tl.sum(tl.sum(k_scaled * k_scaled, axis=0), axis=0)

    absmax_ptr = k_page_absmax + page_id * stride_meta_b + pid_hkv * stride_meta_h
    sumabs_ptr = k_page_sumabs + page_id * stride_meta_b + pid_hkv * stride_meta_h
    l2_ptr = k_page_l2 + page_id * stride_meta_b + pid_hkv * stride_meta_h

    tl.store(absmax_ptr, absmax, mask=page_mask)
    tl.store(sumabs_ptr, sumabs, mask=page_mask)
    tl.store(l2_ptr, tl.sqrt(sumsq), mask=page_mask)


def allocate_paged_kv_cache(
    *,
    max_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: int | None = None,
    k_scale: torch.Tensor,
    v_dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    max_batch: int = 1,
    max_pages_per_seq: int | None = None,
) -> PagedKVCache:
    if max_pages_per_seq is None:
        max_pages_per_seq = max_pages
    vals_per_byte = 4  # q2
    if value_dim is None:
        value_dim = head_dim
    k_packed = (head_dim + vals_per_byte - 1) // vals_per_byte

    k_q = torch.empty(
        (max_pages, num_kv_heads, page_size, k_packed),
        device=device,
        dtype=torch.uint8,
    )
    v = torch.empty(
        (max_pages, num_kv_heads, page_size, value_dim),
        device=device,
        dtype=v_dtype,
    )
    page_lens = torch.full((max_pages,), page_size, device=device, dtype=torch.int32)

    k_page_absmax = torch.zeros((max_pages, num_kv_heads), device=device, dtype=torch.float32)
    k_page_sumabs = torch.zeros((max_pages, num_kv_heads), device=device, dtype=torch.float32)
    k_page_l2 = torch.zeros((max_pages, num_kv_heads), device=device, dtype=torch.float32)

    block_table = torch.full(
        (max_batch, max_pages_per_seq),
        -1,
        device=device,
        dtype=torch.int32,
    )
    page_counts = torch.zeros((max_batch,), device=device, dtype=torch.int32)

    return PagedKVCache(
        k_q=k_q,
        v=v,
        k_scale=k_scale,
        page_lens=page_lens,
        k_page_absmax=k_page_absmax,
        k_page_sumabs=k_page_sumabs,
        k_page_l2=k_page_l2,
        block_table=block_table,
        page_counts=page_counts,
        page_size=page_size,
    )


def update_block_table(
    cache: PagedKVCache,
    seq_ids: Iterable[int],
    page_ids: Iterable[int],
) -> None:
    seq_ids = list(seq_ids)
    page_ids = list(page_ids)
    if len(seq_ids) != len(page_ids):
        raise ValueError("seq_ids and page_ids must have the same length")

    for seq_id, page_id in zip(seq_ids, page_ids):
        count = int(cache.page_counts[seq_id].item())
        cache.block_table[seq_id, count] = int(page_id)
        cache.page_counts[seq_id] = count + 1


def update_pages(
    cache: PagedKVCache,
    page_ids: torch.Tensor,
    k_q_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_lens: Optional[torch.Tensor] = None,
    compute_meta: bool = True,
    block_t: int = 16,
) -> None:
    if page_ids.dim() != 1:
        raise ValueError("page_ids must be a 1D tensor")
    page_ids = page_ids.to(device=cache.k_q.device, dtype=torch.int32).contiguous()
    k_q_pages = k_q_pages.to(device=cache.k_q.device).contiguous()
    v_pages = v_pages.to(device=cache.v.device).contiguous()
    if page_lens is not None:
        page_lens = page_lens.to(device=cache.page_lens.device, dtype=torch.int32).contiguous()
    if k_q_pages.shape != cache.k_q[page_ids].shape:
        raise ValueError("k_q_pages shape must match cache.k_q[page_ids]")
    if v_pages.shape != cache.v[page_ids].shape:
        raise ValueError("v_pages shape must match cache.v[page_ids]")

    cache.k_q[page_ids] = k_q_pages
    cache.v[page_ids] = v_pages
    if page_lens is not None:
        cache.page_lens[page_ids] = page_lens.to(cache.page_lens.dtype)

    if not compute_meta:
        return

    compute_pages_meta(cache, page_ids, block_t=block_t)


def compute_pages_meta(
    cache: PagedKVCache,
    page_ids: torch.Tensor,
    *,
    block_t: int = 16,
) -> None:
    if page_ids.dim() != 1:
        raise ValueError("page_ids must be a 1D tensor")
    page_ids = page_ids.to(device=cache.k_q.device, dtype=torch.int32).contiguous()

    k_scale = cache.k_scale
    if k_scale.dim() == 3:
        if k_scale.shape[0] != 1:
            raise ValueError("k_scale with batch dim must have B=1 for global page cache")
        k_scale = k_scale.squeeze(0)

    grid = (page_ids.numel(), cache.k_q.shape[1])
    compute_page_meta_q2_packed[grid](
        cache.k_q,
        k_scale,
        page_ids,
        cache.page_lens,
        cache.k_page_absmax,
        cache.k_page_sumabs,
        cache.k_page_l2,
        cache.k_q.stride(0),
        cache.k_q.stride(1),
        cache.k_q.stride(2),
        cache.k_q.stride(3),
        k_scale.stride(0),
        k_scale.stride(1),
        cache.k_page_absmax.stride(0),
        cache.k_page_absmax.stride(1),
        page_ids.stride(0),
        cache.page_lens.stride(0),
        cache.k_q.shape[0],
        HKV=cache.k_q.shape[1],
        K=k_scale.shape[1],
        K_PACKED=cache.k_q.shape[-1],
        SBS=cache.page_size,
        BLOCK_T=block_t,
    )


__all__ = [
    "PagedKVCache",
    "allocate_paged_kv_cache",
    "update_block_table",
    "update_pages",
    "compute_pages_meta",
]

# utils/cache.py
"""Utilities for paged KV cache management."""
import json
from pathlib import Path
from typing import Tuple, Optional

import torch

__all__ = [
    "dtype_key", "to_k_str",
    "make_cache_file_path",
    "save_raw_cache", "load_raw_cache",
    "create_paged_kv_cache",
    "convert_to_paged_format",
]


def dtype_key(dt: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dt, str(dt))


def to_k_str(n: int) -> str:
    val = n / 1024.0
    return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"


def make_cache_file_path(raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    raw_dir = Path(raw_data_dir)
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}.json"
    )
    return raw_dir / fname


def save_raw_cache(path, meta: dict, lengths, fused_ms, flash_ms, skip_ratios):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "fused_ms": [float(x) for x in fused_ms],
        "flash_ms": [float(x) for x in flash_ms],
        "skip_ratios": [None if x is None else float(x) for x in skip_ratios],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_raw_cache(path):
    with Path(path).open("r") as f:
        data = json.load(f)
    return (
        data["lengths"],
        data["fused_ms"],
        data["flash_ms"],
        data.get("skip_ratios", [None] * len(data["lengths"])),
        data.get("meta", {}),
    )


def create_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    value_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    k_bits: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create empty paged KV cache tensors.

    Args:
        num_pages: Total number of physical pages
        page_size: Number of tokens per page
        num_kv_heads: Number of KV heads
        head_dim: Dimension of K (and V if value_dim not specified)
        value_dim: Dimension of V (default same as head_dim)
        dtype: Data type for V tensor
        device: Device to create tensors on
        k_bits: Number of bits for quantized K

    Returns:
        Tuple of (k_q, v) where:
            k_q: [num_pages, page_size, num_kv_heads, K_packed] (uint8)
            v: [num_pages, page_size, num_kv_heads, value_dim] (dtype)
    """
    if value_dim is None:
        value_dim = head_dim

    vals_per_byte = 8 // k_bits
    k_packed = (head_dim + vals_per_byte - 1) // vals_per_byte

    k_q = torch.zeros(
        (num_pages, page_size, num_kv_heads, k_packed),
        dtype=torch.uint8,
        device=device
    )
    v = torch.zeros(
        (num_pages, page_size, num_kv_heads, value_dim),
        dtype=dtype,
        device=device
    )

    return k_q, v


def convert_to_paged_format(
    k_q: torch.Tensor,  # [B, T, HKV, K_packed]
    v: torch.Tensor,    # [B, T, HKV, V]
    page_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert contiguous KV cache to paged format.

    Args:
        k_q: Quantized key tensor [B, T, HKV, K_packed]
        v: Value tensor [B, T, HKV, V]
        page_size: Number of tokens per page
        device: Device to create tensors on

    Returns:
        Tuple of (k_q_paged, v_paged, page_table, seq_lens) where:
            k_q_paged: [num_pages, page_size, HKV, K_packed]
            v_paged: [num_pages, page_size, HKV, V]
            page_table: [B, max_pages_per_seq]
            seq_lens: [B]
    """
    B, T, HKV, K_packed = k_q.shape
    _, _, _, V = v.shape

    # Calculate number of pages needed per sequence and total
    pages_per_seq = (T + page_size - 1) // page_size
    num_pages = B * pages_per_seq

    # Create paged tensors
    k_q_paged = torch.zeros(
        (num_pages, page_size, HKV, K_packed),
        dtype=k_q.dtype,
        device=device
    )
    v_paged = torch.zeros(
        (num_pages, page_size, HKV, V),
        dtype=v.dtype,
        device=device
    )

    # Create page table (simple sequential allocation for this conversion)
    page_table = torch.arange(
        0, num_pages,
        dtype=torch.int32,
        device=device
    ).reshape(B, pages_per_seq)

    # Sequence lengths
    seq_lens = torch.full((B,), T, dtype=torch.int32, device=device)

    # Copy data to paged format
    for b in range(B):
        for p in range(pages_per_seq):
            start_t = p * page_size
            end_t = min(start_t + page_size, T)
            actual_len = end_t - start_t

            page_idx = b * pages_per_seq + p
            k_q_paged[page_idx, :actual_len] = k_q[b, start_t:end_t]
            v_paged[page_idx, :actual_len] = v[b, start_t:end_t]

    return k_q_paged, v_paged, page_table, seq_lens


def create_page_table(
    batch_size: int,
    seq_lens: torch.Tensor,
    page_size: int,
    num_pages: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a simple sequential page table allocation.

    Args:
        batch_size: Number of sequences in batch
        seq_lens: Actual sequence lengths [B]
        page_size: Number of tokens per page
        num_pages: Total available pages
        device: Device to create tensor on

    Returns:
        page_table: [B, max_pages_per_seq]
    """
    max_seq_len = int(seq_lens.max().item())
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size

    page_table = torch.zeros(
        (batch_size, max_pages_per_seq),
        dtype=torch.int32,
        device=device
    )

    page_counter = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        pages_needed = (seq_len + page_size - 1) // page_size
        for p in range(pages_needed):
            if page_counter >= num_pages:
                raise RuntimeError(f"Not enough pages: needed {page_counter + 1}, available {num_pages}")
            page_table[b, p] = page_counter
            page_counter += 1

    return page_table

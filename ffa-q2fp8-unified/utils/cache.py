# utils/cache.py
import json
from pathlib import Path

import torch

__all__ = [
    "dtype_key", "to_k_str",
    "make_cache_file_path",
    "save_raw_cache", "load_raw_cache"
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

def make_cache_file_path(raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1, current_len=0, cudagraph=False, replay_only=False):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    raw_dir = Path(raw_data_dir)
    suffix = ""
    if cudagraph:
        suffix = "_cudagraph_replay" if replay_only else "_cudagraph"
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}_curr{current_len}{suffix}.json"
    )
    return raw_dir / fname

def save_raw_cache(path, meta: dict, lengths, q2_ms, q2_cg_ms_or_flash_ms, flash_ms=None, skip_ratios=None):
    """Save benchmark results to cache.

    Args:
        path: Cache file path
        meta: Metadata dict
        lengths: List of sequence lengths
        q2_ms: Q2FP8 baseline timings (can be None for each entry)
        q2_cg_ms_or_flash_ms: Either Q2FP8 CUDAGraph timings OR FlashAttention timings (for non-CG mode)
        flash_ms: FlashAttention timings (optional, for CUDAGraph mode)
        skip_ratios: Skip ratios (optional)
    """
    path = Path(path)

    # Determine if this is CUDAGraph mode based on meta or presence of flash_ms
    is_cudagraph = meta.get("cudagraph", False) or flash_ms is not None

    if is_cudagraph:
        # CUDAGraph mode: q2_ms, q2_cg_ms, flash_ms
        payload = {
            "meta": meta,
            "lengths": [int(x) for x in lengths],
            "q2_ms": [None if x is None else float(x) for x in q2_ms],
            "q2_cg_ms": [float(x) for x in q2_cg_ms_or_flash_ms],
            "flash_ms": [None if x is None else float(x) for x in (flash_ms or [None] * len(lengths))],
            "skip_ratios": [None if x is None else float(x) for x in (skip_ratios or [None] * len(lengths))],
        }
    else:
        # Non-CUDAGraph mode: unified_ms, flash_ms
        payload = {
            "meta": meta,
            "lengths": [int(x) for x in lengths],
            "unified_ms": [float(x) for x in q2_ms],
            "flash_ms": [None if x is None else float(x) for x in q2_cg_ms_or_flash_ms],
            "skip_ratios": [None if x is None else float(x) for x in (skip_ratios or [None] * len(lengths))],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)

def load_raw_cache(path):
    """Load benchmark results from cache.

    Returns:
        For CUDAGraph mode: (lengths, q2_ms, q2_cg_ms, flash_ms, skip_ratios, meta)
        For non-CUDAGraph mode: (lengths, unified_ms, flash_ms, skip_ratios, meta)
    """
    with Path(path).open("r") as f:
        data = json.load(f)

    # Check if this is CUDAGraph mode
    if "q2_cg_ms" in data:
        # CUDAGraph mode
        return (
            data["lengths"],
            data.get("q2_ms", [None] * len(data["lengths"])),
            data["q2_cg_ms"],
            data.get("flash_ms", [None] * len(data["lengths"])),
            data.get("skip_ratios", [None] * len(data["lengths"])),
            data.get("meta", {}),
        )
    else:
        # Non-CUDAGraph mode (backward compatible)
        return (
            data["lengths"],
            data["unified_ms"],
            data.get("flash_ms", [None] * len(data["lengths"])),
            data.get("skip_ratios", [None] * len(data["lengths"])),
            data.get("meta", {}),
        )

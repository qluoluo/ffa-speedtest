import json
from pathlib import Path

import torch

__all__ = [
    "dtype_key",
    "to_k_str",
    "make_cache_file_path",
    "save_meta_cache",
    "load_meta_cache",
    "save_attn_cache",
    "load_attn_cache",
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


def make_cache_file_path(raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, SBS, dtype, step, iters, warmup, bsz=1):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    raw_dir = Path(raw_data_dir)
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_SBS{SBS}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}.json"
    )
    return raw_dir / fname


def save_meta_cache(path, meta: dict, lengths, update_ms, meta_ms, flash_ms=None):
    path = Path(path)
    if flash_ms is None:
        flash_ms = [None] * len(lengths)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "update_ms": [None if x is None else float(x) for x in update_ms],
        "meta_ms": [None if x is None else float(x) for x in meta_ms],
        "flash_ms": [None if x is None else float(x) for x in flash_ms],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_meta_cache(path):
    with Path(path).open("r") as f:
        data = json.load(f)
    return (
        data["lengths"],
        data.get("update_ms", [None] * len(data["lengths"])),
        data.get("meta_ms", [None] * len(data["lengths"])),
        data.get("flash_ms", [None] * len(data["lengths"])),
        data.get("meta", {}),
    )


def save_attn_cache(path, meta: dict, lengths, paged_ms, flash_ms):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "paged_ms": [None if x is None else float(x) for x in paged_ms],
        "flash_ms": [None if x is None else float(x) for x in flash_ms],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_attn_cache(path):
    with Path(path).open("r") as f:
        data = json.load(f)
    return (
        data["lengths"],
        data.get("paged_ms", [None] * len(data["lengths"])),
        data.get("flash_ms", [None] * len(data["lengths"])),
        data.get("meta", {}),
    )

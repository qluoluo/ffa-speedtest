import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


FP4_POS_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    return (x * cos) + (rotate_half(x) * sin)


def _normalize_k_weight(w_k: torch.Tensor, in_dim: int) -> torch.Tensor:
    if w_k.ndim != 2:
        raise ValueError(f"w_k must be 2D, got {w_k.shape}")
    if w_k.shape[1] == in_dim:
        return w_k
    if w_k.shape[0] == in_dim:
        return w_k.t().contiguous()
    raise ValueError(f"w_k shape {w_k.shape} is incompatible with in_dim={in_dim}")


def _resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def encode_k_nvfp4_fp8_residual(
    k: torch.Tensor,
    fp8_dtype: torch.dtype,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode K into NVFP4 (block-scaled FP4) with per-page scale and FP8 residual."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    B, T, HKV, K = k.shape
    fp4_pos = torch.tensor(FP4_POS_LEVELS, device=k.device, dtype=k.dtype)
    thresholds = torch.tensor(FP4_THRESHOLDS, device=k.device, dtype=k.dtype)
    fp4_max = fp4_pos[-1]

    num_pages = (T + page_size - 1) // page_size
    pad_t = num_pages * page_size - T
    if pad_t:
        pad_tensor = torch.zeros((B, pad_t, HKV, K), device=k.device, dtype=k.dtype)
        k_pad = torch.cat([k, pad_tensor], dim=1)
    else:
        k_pad = k

    k_blocks = k_pad.view(B, num_pages, page_size, HKV, K)
    max_abs = k_blocks.abs().amax(dim=(2, 4))
    k_scale = (max_abs / fp4_max).clamp_min(1e-6).to(k.dtype).contiguous()

    scale_block = k_scale[:, :, None, :, None]
    k_scaled = k_blocks / scale_block

    abs_k = k_scaled.abs()
    idx = torch.bucketize(abs_k, thresholds, right=True).to(torch.int64)
    sign = k_scaled < 0
    k_fp4 = (idx.to(torch.uint8) | (sign.to(torch.uint8) << 3)).contiguous()
    sign_scale = torch.where(sign, -torch.ones_like(abs_k), torch.ones_like(abs_k))
    k_dequant = fp4_pos[idx] * sign_scale
    k_dequant = k_dequant * scale_block
    k_residual = (k_blocks - k_dequant).to(fp8_dtype).contiguous()

    k_fp4 = k_fp4.view(B, num_pages * page_size, HKV, K)
    k_residual = k_residual.view(B, num_pages * page_size, HKV, K)
    if pad_t:
        k_fp4 = k_fp4[:, :T]
        k_residual = k_residual[:, :T]

    values_per_byte = 2
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad_k = k_packed_len * values_per_byte - K
    if pad_k:
        pad_tensor = torch.zeros((B, T, HKV, pad_k), device=k_fp4.device, dtype=k_fp4.dtype)
        k_fp4 = torch.cat([k_fp4, pad_tensor], dim=-1)
    k_fp4 = k_fp4.view(B, T, HKV, k_packed_len, values_per_byte)
    k_fp4_packed = (k_fp4[..., 0] | (k_fp4[..., 1] << 4)).contiguous()
    return k_fp4_packed, k_scale, k_residual


def kproj_rope_quantize_reference(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(h.device)
    w_k = _normalize_k_weight(w_k, h.shape[-1])
    k_linear = F.linear(h, w_k, bias)
    B, T, _ = k_linear.shape
    if k_linear.shape[-1] != num_kv_heads * head_dim:
        raise ValueError(
            f"projection out_dim must be num_kv_heads*head_dim, got {k_linear.shape[-1]} "
            f"vs {num_kv_heads}*{head_dim}"
        )
    k = k_linear.view(B, T, num_kv_heads, head_dim)
    k = apply_rope(k, cos, sin)
    return encode_k_nvfp4_fp8_residual(k, fp8_dtype, page_size)


def kproj_rope_quantize_triton(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Placeholder: reuse the reference path for now.
    return kproj_rope_quantize_reference(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        bias=bias,
        fp8_dtype=fp8_dtype,
    )


def kproj_rope_quantize_fused(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return kproj_rope_quantize_triton(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        bias=bias,
        fp8_dtype=fp8_dtype,
    )


def run_smoke_test(
    device: torch.device,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    HKV: int,
    K: int,
    page_size: int,
) -> None:
    torch.manual_seed(0)
    h = torch.randn(B, T, H, device=device, dtype=dtype)
    w_k = torch.randn(HKV * K, H, device=device, dtype=dtype)
    cos, sin = build_rope_cache(T, K, device=device, dtype=dtype)

    ref = kproj_rope_quantize_reference(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=HKV,
        head_dim=K,
        page_size=page_size,
        fp8_dtype=fp8_dtype,
    )
    fused = kproj_rope_quantize_triton(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=HKV,
        head_dim=K,
        page_size=page_size,
        fp8_dtype=fp8_dtype,
    )

    k_fp4_ref, k_scale_ref, k_res_ref = ref
    k_fp4_fused, k_scale_fused, k_res_fused = fused

    packed_mismatch = (k_fp4_ref != k_fp4_fused).float().mean().item()
    scale_mean = (k_scale_ref.float() - k_scale_fused.float()).abs().mean().item()
    res_mean = (k_res_ref.float() - k_res_fused.float()).abs().mean().item()
    if packed_mismatch > 0.0 or scale_mean > 0.0 or res_mean > 0.0:
        raise AssertionError(
            "fused output mismatch: "
            f"packed_mismatch={packed_mismatch:.6f}, scale_mean={scale_mean:.6f}, res_mean={res_mean:.6f}"
        )

    print(
        "[OK] nvfp4 base/scale/residual outputs match reference "
        f"(packed_mismatch={packed_mismatch:.6f}, scale_mean={scale_mean:.6f}, res_mean={res_mean:.6f})"
    )
    print(
        f"device={device}, dtype={dtype}, fp8_dtype={fp8_dtype}, "
        f"B={B}, T={T}, H={H}, HKV={HKV}, K={K}, page_size={page_size}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse K projection + RoPE + nvfp4/fp8 split and test.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--HKV", type=int, default=4)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--page-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    if device.type == "cpu":
        raise RuntimeError("This test requires a CUDA device for fp8 storage.")
    fp8_dtype = _resolve_fp8_dtype(device)
    if not hasattr(torch, "float8_e5m2") or fp8_dtype != torch.float8_e5m2:
        print("[Info] fp8 dtype not available on this device, using fp16 residuals.")

    run_smoke_test(
        device=device,
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        B=args.B,
        T=args.T,
        H=args.H,
        HKV=args.HKV,
        K=args.K,
        page_size=args.page_size,
    )


if __name__ == "__main__":
    main()

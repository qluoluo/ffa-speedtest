"""
Page-based Q2FP8 quantization kernels.

Key difference from original ffa-q2fp8-threshold:
- Original: quantize across entire sequence T, scale/zero are [B, HKV, K]
- Paged: quantize per page independently, scale/zero are per-page [HKV, K]
"""

from typing import Optional, Tuple

import torch


def _resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    """Resolve FP8 dtype, fallback to FP16 if not supported."""
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def quantize_k_page_q2fp8(
    k_page: torch.Tensor,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a single page of K cache to 2-bit + FP8 residual.

    Args:
        k_page: [HKV, PAGE_SIZE, K] or [B, HKV, PAGE_SIZE, K]
            Single page of K cache (FP16/BF16/FP32)
        fp8_dtype: FP8 dtype for residual (default: float8_e5m2 if available, else fp16)

    Returns:
        k_q_packed: [HKV, PAGE_SIZE, K_packed] or [B, HKV, PAGE_SIZE, K_packed]
            Packed 2-bit values (uint8), 4 values per byte
        k_scale: [HKV, K] or [B, HKV, K]
            Per-channel quantization scale
        k_zero: [HKV, K] or [B, HKV, K]
            Per-channel quantization zero point
        k_residual: [HKV, PAGE_SIZE, K] or [B, HKV, PAGE_SIZE, K]
            FP8 residual

    Notes:
        - Quantization is per-channel (shared across PAGE_SIZE tokens)
        - 2-bit quantization: values in {0, 1, 2, 3}
        - scale = (max - min) / 3
        - quantized = round((k - zero) / scale)
        - residual = k - dequantized
    """
    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(k_page.device)

    # Support both [HKV, PAGE_SIZE, K] and [B, HKV, PAGE_SIZE, K]
    has_batch = k_page.ndim == 4
    if not has_batch:
        k_page = k_page.unsqueeze(0)  # Add batch dim

    B, HKV, PAGE_SIZE, K = k_page.shape

    # 1. Compute per-channel min/max (across PAGE_SIZE tokens)
    # k_min, k_max: [B, HKV, K]
    k_min = k_page.amin(dim=2)  # min over PAGE_SIZE
    k_max = k_page.amax(dim=2)  # max over PAGE_SIZE

    # 2. Compute scale and zero point
    # 2-bit has 4 levels (0, 1, 2, 3), so scale = (max - min) / 3
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()

    # 3. Quantize to 2-bit (values in [0, 3])
    # k_q: [B, HKV, PAGE_SIZE, K]
    k_q = torch.round(
        (k_page - zero[:, :, None, :]) / scale[:, :, None, :]
    ).clamp(0, 3).to(torch.uint8)

    # 4. Compute dequantized values and residual
    k_dequant = (
        k_q.to(torch.float32) * scale[:, :, None, :].to(torch.float32)
        + zero[:, :, None, :].to(torch.float32)
    )
    k_residual = (k_page.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # 5. Pack 4 x 2-bit values into 1 byte
    # Padding to multiple of 4
    values_per_byte = 4
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad > 0:
        pad_tensor = torch.zeros(
            (B, HKV, PAGE_SIZE, pad),
            device=k_q.device,
            dtype=k_q.dtype
        )
        k_q = torch.cat([k_q, pad_tensor], dim=-1)

    # Reshape and pack: [B, HKV, PAGE_SIZE, K_packed, 4]
    k_q = k_q.view(B, HKV, PAGE_SIZE, k_packed_len, values_per_byte)

    # Bit packing: value0 | (value1 << 2) | (value2 << 4) | (value3 << 6)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()

    # Remove batch dim if input didn't have it
    if not has_batch:
        k_q_packed = k_q_packed.squeeze(0)
        scale = scale.squeeze(0)
        zero = zero.squeeze(0)
        k_residual = k_residual.squeeze(0)

    return k_q_packed, scale, zero, k_residual


def dequantize_k_page_q2fp8(
    k_q_packed: torch.Tensor,
    k_scale: torch.Tensor,
    k_zero: torch.Tensor,
    k_residual: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Dequantize a page of K cache from 2-bit + FP8 residual.

    Args:
        k_q_packed: [HKV, PAGE_SIZE, K_packed] or [B, HKV, PAGE_SIZE, K_packed]
            Packed 2-bit values
        k_scale: [HKV, K] or [B, HKV, K]
            Per-channel scale
        k_zero: [HKV, K] or [B, HKV, K]
            Per-channel zero point
        k_residual: [HKV, PAGE_SIZE, K] or [B, HKV, PAGE_SIZE, K], optional
            FP8 residual
        head_dim: int, optional
            Original head dimension K (required if packed)

    Returns:
        k_page: [HKV, PAGE_SIZE, K] or [B, HKV, PAGE_SIZE, K]
            Dequantized K cache
    """
    has_batch = k_q_packed.ndim == 4
    if not has_batch:
        k_q_packed = k_q_packed.unsqueeze(0)
        k_scale = k_scale.unsqueeze(0)
        k_zero = k_zero.unsqueeze(0)
        if k_residual is not None:
            k_residual = k_residual.unsqueeze(0)

    B, HKV, PAGE_SIZE, K_packed = k_q_packed.shape

    # Infer head_dim if not provided
    if head_dim is None:
        head_dim = k_scale.shape[-1]

    # 1. Unpack 2-bit values
    # Extract 4 x 2-bit values from each byte
    k_q = torch.stack([
        k_q_packed & 0x3,           # bits 0-1
        (k_q_packed >> 2) & 0x3,    # bits 2-3
        (k_q_packed >> 4) & 0x3,    # bits 4-5
        (k_q_packed >> 6) & 0x3,    # bits 6-7
    ], dim=-1)  # [B, HKV, PAGE_SIZE, K_packed, 4]

    # Flatten: [B, HKV, PAGE_SIZE, K_packed * 4]
    k_q = k_q.view(B, HKV, PAGE_SIZE, K_packed * 4)

    # Trim padding
    k_q = k_q[..., :head_dim]

    # 2. Dequantize: k = k_q * scale + zero
    k_page = (
        k_q.to(k_scale.dtype) * k_scale[:, :, None, :]
        + k_zero[:, :, None, :]
    )

    # 3. Add residual if provided
    if k_residual is not None:
        k_page = k_page + k_residual.to(k_page.dtype)

    # Remove batch dim if input didn't have it
    if not has_batch:
        k_page = k_page.squeeze(0)

    return k_page


def quantize_k_multi_pages(
    k: torch.Tensor,
    page_size: int,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize K cache with multiple pages.

    Args:
        k: [B, T, HKV, K]
            K cache (FP16/BF16/FP32)
        page_size: int
            Tokens per page
        fp8_dtype: FP8 dtype for residual

    Returns:
        k_q_packed: [B, NUM_PAGES, HKV, PAGE_SIZE, K_packed]
        k_scale: [B, NUM_PAGES, HKV, K]
        k_zero: [B, NUM_PAGES, HKV, K]
        k_residual: [B, NUM_PAGES, HKV, PAGE_SIZE, K]

    Notes:
        - Pads sequence length to multiple of page_size
        - Each page is quantized independently
    """
    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(k.device)

    B, T, HKV, K = k.shape

    # Pad to multiple of page_size
    num_pages = (T + page_size - 1) // page_size
    padded_T = num_pages * page_size
    if padded_T > T:
        pad_len = padded_T - T
        k = torch.cat([
            k,
            torch.zeros((B, pad_len, HKV, K), device=k.device, dtype=k.dtype)
        ], dim=1)

    # Reshape to pages: [B, NUM_PAGES, PAGE_SIZE, HKV, K]
    k_pages = k.view(B, num_pages, page_size, HKV, K)
    k_pages = k_pages.transpose(2, 3)  # [B, NUM_PAGES, HKV, PAGE_SIZE, K]

    # Quantize each page
    all_q_packed = []
    all_scale = []
    all_zero = []
    all_residual = []

    for page_idx in range(num_pages):
        k_page = k_pages[:, page_idx]  # [B, HKV, PAGE_SIZE, K]
        q_packed, scale, zero, residual = quantize_k_page_q2fp8(k_page, fp8_dtype)
        all_q_packed.append(q_packed)
        all_scale.append(scale)
        all_zero.append(zero)
        all_residual.append(residual)

    # Stack: [B, NUM_PAGES, ...]
    k_q_packed = torch.stack(all_q_packed, dim=1)
    k_scale = torch.stack(all_scale, dim=1)
    k_zero = torch.stack(all_zero, dim=1)
    k_residual = torch.stack(all_residual, dim=1)

    return k_q_packed, k_scale, k_zero, k_residual


if __name__ == "__main__":
    # Test quantization
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single page test
    HKV = 8
    PAGE_SIZE = 128
    K = 128

    k_page = torch.randn(HKV, PAGE_SIZE, K, device=device, dtype=torch.float16)

    print(f"Input shape: {k_page.shape}")

    # Quantize
    k_q_packed, k_scale, k_zero, k_residual = quantize_k_page_q2fp8(k_page)

    print(f"Quantized packed shape: {k_q_packed.shape}")
    print(f"Scale shape: {k_scale.shape}")
    print(f"Zero shape: {k_zero.shape}")
    print(f"Residual shape: {k_residual.shape}")

    # Dequantize
    k_page_recon = dequantize_k_page_q2fp8(k_q_packed, k_scale, k_zero, k_residual, head_dim=K)

    print(f"Reconstructed shape: {k_page_recon.shape}")

    # Check error
    error = (k_page - k_page_recon).abs()
    print(f"Max error: {error.max().item():.6f}")
    print(f"Mean error: {error.mean().item():.6f}")

    # Multi-page test
    B = 2
    T = 384  # Will be split into 3 pages
    k_multi = torch.randn(B, T, HKV, K, device=device, dtype=torch.float16)

    print(f"\nMulti-page input shape: {k_multi.shape}")

    k_q_packed_multi, k_scale_multi, k_zero_multi, k_residual_multi = quantize_k_multi_pages(
        k_multi, page_size=PAGE_SIZE
    )

    print(f"Multi-page quantized shape: {k_q_packed_multi.shape}")
    print(f"Multi-page scale shape: {k_scale_multi.shape}")

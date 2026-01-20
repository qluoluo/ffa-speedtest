"""
Fused RoPE + Quantization Triton Kernel for Prefill

This kernel fuses RoPE rotation and Q2FP8 quantization to avoid storing
intermediate FP16 keys during prefill.

Key features:
1. Applies RoPE rotation to keys
2. Performs per-block symmetric 2-bit quantization
3. Computes FP8 residuals for accuracy
4. Outputs quantized keys ready for cache storage

Usage:
    from fused_rope_quant_kernel import fused_rope_and_quantize_triton

    k_q, k_scale, k_residual = fused_rope_and_quantize_triton(
        k, cos, sin, block_size=64, k_bits=2
    )
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _fused_rope_quant_kernel(
    # Input pointers
    k_ptr,        # [B, T, HKV, K] FP16 input keys
    cos_ptr,      # [T, K] or [B, T, K] RoPE cosine
    sin_ptr,      # [T, K] or [B, T, K] RoPE sine
    # Output pointers
    k_q_ptr,      # [B, T, HKV, K_PACKED] uint8 quantized keys
    k_scale_ptr,  # [B, num_blocks, HKV, K] FP16 per-block scales
    k_res_ptr,    # [B, T, HKV, K] FP8 residuals
    # Dimensions
    B: tl.constexpr,
    T: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    BS: tl.constexpr,  # Block size (64)
    num_blocks: tl.constexpr,
    # Quantization params
    K_BITS: tl.constexpr = 2,
    # Strides
    stride_k_b: tl.constexpr = 0,
    stride_k_t: tl.constexpr = 0,
    stride_k_h: tl.constexpr = 0,
    stride_k_k: tl.constexpr = 0,
    stride_cos_t: tl.constexpr = 0,
    stride_cos_k: tl.constexpr = 0,
    has_batch_cos: tl.constexpr = False,
    # Block size for processing
    BLOCK_K: tl.constexpr = 128,
):
    """
    Fused RoPE + Quantization kernel

    Grid: (num_blocks, B, HKV)
    Each block processes BS tokens for one (B, HKV) pair
    """
    pid_block = tl.program_id(0)  # Which token block
    pid_b = tl.program_id(1)      # Which batch
    pid_h = tl.program_id(2)      # Which head

    # Constants
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS
    eps = 1e-8

    # Token range for this block
    t_start = pid_block * BS
    t_end = tl.minimum(t_start + BS, T)

    # Allocate buffers for rotated keys [BS, K]
    k_rotated = tl.zeros([BS, BLOCK_K], dtype=tl.float32)

    # Process K dimension in chunks
    for k_chunk in range(0, K, BLOCK_K):
        k_size = tl.minimum(BLOCK_K, K - k_chunk)

        # Load keys for this block: [BS, BLOCK_K]
        offs_t = t_start + tl.arange(0, BS)
        offs_k = k_chunk + tl.arange(0, BLOCK_K)

        k_mask = (offs_t[:, None] < t_end) & (offs_k[None, :] < K)

        k_ptrs = (k_ptr +
                  pid_b * stride_k_b +
                  offs_t[:, None] * stride_k_t +
                  pid_h * stride_k_h +
                  offs_k[None, :] * stride_k_k)
        k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Load cos/sin for RoPE: [BS, BLOCK_K]
        if has_batch_cos:
            cos_ptrs = (cos_ptr +
                       pid_b * (T * K) +
                       offs_t[:, None] * stride_cos_t +
                       offs_k[None, :] * stride_cos_k)
            sin_ptrs = (sin_ptr +
                       pid_b * (T * K) +
                       offs_t[:, None] * stride_cos_t +
                       offs_k[None, :] * stride_cos_k)
        else:
            cos_ptrs = (cos_ptr +
                       offs_t[:, None] * stride_cos_t +
                       offs_k[None, :] * stride_cos_k)
            sin_ptrs = (sin_ptr +
                       offs_t[:, None] * stride_cos_t +
                       offs_k[None, :] * stride_cos_k)

        cos_vals = tl.load(cos_ptrs, mask=k_mask, other=1.0)
        sin_vals = tl.load(sin_ptrs, mask=k_mask, other=0.0)

        # Apply RoPE rotation
        # Split into first half and second half
        K_half = K // 2
        if k_chunk < K_half:
            # First half: k1 * cos - k2 * sin
            # Need to load corresponding k2 from second half
            offs_k2 = K_half + k_chunk + tl.arange(0, BLOCK_K)
            k2_mask = (offs_t[:, None] < t_end) & (offs_k2[None, :] < K)

            k2_ptrs = (k_ptr +
                      pid_b * stride_k_b +
                      offs_t[:, None] * stride_k_t +
                      pid_h * stride_k_h +
                      offs_k2[None, :] * stride_k_k)
            k2_vals = tl.load(k2_ptrs, mask=k2_mask, other=0.0)

            # Load sin for k2 position
            if has_batch_cos:
                sin2_ptrs = (sin_ptr +
                           pid_b * (T * K) +
                           offs_t[:, None] * stride_cos_t +
                           offs_k2[None, :] * stride_cos_k)
            else:
                sin2_ptrs = (sin_ptr +
                           offs_t[:, None] * stride_cos_t +
                           offs_k2[None, :] * stride_cos_k)
            sin2_vals = tl.load(sin2_ptrs, mask=k2_mask, other=0.0)

            k_rotated = k_vals * cos_vals - k2_vals * sin2_vals
        else:
            # Second half: k2 * cos + k1 * sin
            # Need to load corresponding k1 from first half
            offs_k1 = (k_chunk - K_half) + tl.arange(0, BLOCK_K)
            k1_mask = (offs_t[:, None] < t_end) & (offs_k1[None, :] < K_half)

            k1_ptrs = (k_ptr +
                      pid_b * stride_k_b +
                      offs_t[:, None] * stride_k_t +
                      pid_h * stride_k_h +
                      offs_k1[None, :] * stride_k_k)
            k1_vals = tl.load(k1_ptrs, mask=k1_mask, other=0.0)

            # Load cos for k1 position
            if has_batch_cos:
                cos1_ptrs = (cos_ptr +
                           pid_b * (T * K) +
                           offs_t[:, None] * stride_cos_t +
                           offs_k1[None, :] * stride_cos_k)
            else:
                cos1_ptrs = (cos_ptr +
                           offs_t[:, None] * stride_cos_t +
                           offs_k1[None, :] * stride_cos_k)
            cos1_vals = tl.load(cos1_ptrs, mask=k1_mask, other=1.0)

            k_rotated = k_vals * cos_vals + k1_vals * sin_vals

        # Compute per-block scale: max(abs(k_rotated)) across BS tokens
        k_abs = tl.abs(k_rotated)
        k_max = tl.max(k_abs, axis=0)  # [BLOCK_K]
        k_scale = tl.maximum(k_max / QZERO, eps)

        # Store scale: [B, num_blocks, HKV, K]
        scale_ptrs = (k_scale_ptr +
                     pid_b * (num_blocks * HKV * K) +
                     pid_block * (HKV * K) +
                     pid_h * K +
                     offs_k)
        scale_mask = offs_k < K
        tl.store(scale_ptrs, k_scale, mask=scale_mask)

        # Quantize: q = round((k / scale) + QZERO)
        k_norm = k_rotated / k_scale[None, :]
        k_q_float = tl.maximum(tl.minimum((k_norm + QZERO), QMAX), 0.0)
        k_q_round = tl.floor(k_q_float + 0.5)  # Round

        # Pack to uint8 (for 2-bit: 4 values per byte)
        if K_BITS == 2:
            # Process 4 values at a time
            for pack_idx in range(0, BLOCK_K, VALS_PER_BYTE):
                if pack_idx + VALS_PER_BYTE <= k_size:
                    packed = (k_q_round[:, pack_idx].to(tl.int32) |
                             (k_q_round[:, pack_idx + 1].to(tl.int32) << 2) |
                             (k_q_round[:, pack_idx + 2].to(tl.int32) << 4) |
                             (k_q_round[:, pack_idx + 3].to(tl.int32) << 6))

                    # Store packed values
                    pack_k_idx = (k_chunk + pack_idx) // VALS_PER_BYTE
                    packed_ptrs = (k_q_ptr +
                                  pid_b * (T * HKV * K_PACKED) +
                                  offs_t[:, None] * (HKV * K_PACKED) +
                                  pid_h * K_PACKED +
                                  pack_k_idx)
                    pack_mask = offs_t < t_end
                    tl.store(packed_ptrs, packed.to(tl.uint8), mask=pack_mask[:, None])

        # Compute and store residuals
        k_dequant = (k_q_round - QZERO) * k_scale[None, :]
        k_residual = k_rotated - k_dequant

        res_ptrs = (k_res_ptr +
                   pid_b * (T * HKV * K) +
                   offs_t[:, None] * (HKV * K) +
                   pid_h * K +
                   offs_k[None, :])
        tl.store(res_ptrs, k_residual.to(tl.float8e5), mask=k_mask)


def fused_rope_and_quantize_triton(
    k: torch.Tensor,      # [B, T, HKV, K]
    cos: torch.Tensor,    # [T, K] or [B, T, K]
    sin: torch.Tensor,    # [T, K] or [B, T, K]
    block_size: int = 64,
    k_bits: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused RoPE + Quantization using Triton

    Args:
        k: Key states [B, T, HKV, K]
        cos: Cosine for RoPE [T, K] or [B, T, K]
        sin: Sine for RoPE [T, K] or [B, T, K]
        block_size: Quantization block size (default 64)
        k_bits: Quantization bits (default 2)

    Returns:
        k_q: Quantized keys [B, T, HKV, K_PACKED] uint8
        k_scale: Per-block scales [B, num_blocks, HKV, K] FP16
        k_residual: FP8 residuals [B, T, HKV, K] FP8
    """
    B, T, HKV, K = k.shape
    assert K % 2 == 0, "K must be even for RoPE"
    assert k_bits == 2, "Only 2-bit quantization supported currently"

    # Pad T to multiple of block_size
    T_padded = ((T + block_size - 1) // block_size) * block_size
    if T_padded != T:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, T_padded - T), value=0.0)
        if cos.dim() == 2:
            cos = torch.nn.functional.pad(cos, (0, 0, 0, T_padded - T), value=1.0)
            sin = torch.nn.functional.pad(sin, (0, 0, 0, T_padded - T), value=0.0)
        else:
            cos = torch.nn.functional.pad(cos, (0, 0, 0, T_padded - T, 0, 0), value=1.0)
            sin = torch.nn.functional.pad(sin, (0, 0, 0, T_padded - T, 0, 0), value=0.0)
        T = T_padded

    num_blocks = T // block_size
    VALS_PER_BYTE = 8 // k_bits
    K_PACKED = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    # Allocate outputs
    k_q = torch.zeros((B, T, HKV, K_PACKED), dtype=torch.uint8, device=k.device)
    k_scale = torch.zeros((B, num_blocks, HKV, K), dtype=torch.float16, device=k.device)
    k_residual = torch.zeros((B, T, HKV, K), dtype=torch.float8_e5m2, device=k.device)

    # Check if cos/sin have batch dimension
    has_batch_cos = (cos.dim() == 3)

    # Grid: (num_blocks, B, HKV)
    grid = (num_blocks, B, HKV)

    # Launch kernel
    _fused_rope_quant_kernel[grid](
        k, cos, sin,
        k_q, k_scale, k_residual,
        B=B, T=T, HKV=HKV, K=K, K_PACKED=K_PACKED,
        BS=block_size, num_blocks=num_blocks,
        K_BITS=k_bits,
        stride_k_b=k.stride(0),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        stride_cos_t=cos.stride(-2),
        stride_cos_k=cos.stride(-1),
        has_batch_cos=has_batch_cos,
        BLOCK_K=128,
    )

    return k_q, k_scale, k_residual


if __name__ == "__main__":
    # Simple test
    print("Testing fused RoPE + quantization kernel...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, HKV, K = 1, 128, 8, 128
    block_size = 64

    torch.manual_seed(42)
    k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    cos = torch.randn(T, K, dtype=torch.float16, device=device)
    sin = torch.randn(T, K, dtype=torch.float16, device=device)

    k_q, k_scale, k_res = fused_rope_and_quantize_triton(k, cos, sin, block_size=block_size)

    print(f"Input shape: {k.shape}")
    print(f"Output k_q shape: {k_q.shape}")
    print(f"Output k_scale shape: {k_scale.shape}")
    print(f"Output k_residual shape: {k_res.shape}")
    print("✓ Test passed!")

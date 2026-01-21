"""
Fused RoPE + Quantization Kernel

将 RoPE 和 KV cache 量化合并成一个 kernel，减少内存访问和中间结果存储。

优化点：
1. K tensor 只读一次，直接应用 RoPE 后量化
2. 避免中间结果写回 global memory
3. 减少 transpose 操作
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def fused_rope_quantize_kernel(
    # Input
    k_ptr,  # [B, T, HKV, K] - input key states
    cos_ptr,  # [B, T, K] or [1, T, K] - cos for RoPE
    sin_ptr,  # [B, T, K] or [1, T, K] - sin for RoPE
    # Output
    k_q_ptr,  # [B, T, HKV, K_packed] - quantized output
    k_scale_ptr,  # [B, num_blocks, HKV, K] - scales per block
    k_residual_ptr,  # [B, T, HKV, K] - FP8 residual
    # Dimensions
    B: tl.constexpr,
    T: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    BS: tl.constexpr,  # block size for quantization
    K_BITS: tl.constexpr,  # 2 or 4
    # Strides
    stride_kb, stride_kt, stride_kh, stride_kk,
    stride_cosb, stride_cost, stride_cosk,
    stride_sinb, stride_sint, stride_sink,
    stride_qb, stride_qt, stride_qh, stride_qk,
    stride_sb, stride_sblk, stride_sh, stride_sk,
    stride_rb, stride_rt, stride_rh, stride_rk,
):
    """
    Fused RoPE + Quantization kernel

    Grid: (num_blocks, HKV, B)
    Each block processes BS tokens for one head in one batch
    """
    # Program IDs
    pid_block = tl.program_id(0)  # which block (T // BS)
    pid_h = tl.program_id(1)      # which head
    pid_b = tl.program_id(2)      # which batch

    # Constants
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS
    EPS = 1e-8
    K_HALF: tl.constexpr = K // 2

    # Token range for this block
    t_start = pid_block * BS

    # Offsets
    offs_k = tl.arange(0, K)
    offs_k1 = tl.arange(0, K_HALF)
    offs_k2 = K_HALF + tl.arange(0, K_HALF)

    # Base pointers
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    cos_base = cos_ptr + pid_b * stride_cosb
    sin_base = sin_ptr + pid_b * stride_sinb

    # Accumulate max for scale computation
    k_max = tl.zeros([K], dtype=tl.float32) + EPS

    # Process each token in the block and apply RoPE
    k_rotated = tl.zeros([BS, K], dtype=tl.float32)

    for t_idx in range(BS):
        t_offset = t_start + t_idx
        if t_offset < T:
            # Load K for this token
            k_ptrs = k_base + t_offset * stride_kt + offs_k * stride_kk
            k_vals = tl.load(k_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)

            # Load cos and sin
            cos_ptrs = cos_base + t_offset * stride_cost + offs_k * stride_cosk
            sin_ptrs = sin_base + t_offset * stride_sint + offs_k * stride_sink
            cos_vals = tl.load(cos_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
            sin_vals = tl.load(sin_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)

            # Apply RoPE: k_embed = k * cos + rotate_half(k) * sin
            # rotate_half: [-k[K/2:], k[:K/2]]

            # Create masks for two halves
            mask_first_half = offs_k < K_HALF
            mask_second_half = offs_k >= K_HALF

            # Build rotate_half vector:
            # - First half (indices 0 to K_HALF-1): -k_vals[K_HALF:K]
            # - Second half (indices K_HALF to K-1): k_vals[0:K_HALF]

            # For first half: we need k_vals[offs_k + K_HALF]
            # For second half: we need k_vals[offs_k - K_HALF]
            k_rot_half = tl.where(
                mask_first_half,
                -k_vals,  # Will be shifted below
                k_vals    # Will be shifted below
            )

            # Actually, we need to use tl.load with different offsets
            # Let's reload with shifted indices
            k_rot_ptrs_first = k_base + t_offset * stride_kt + (offs_k + K_HALF) * stride_kk
            k_rot_ptrs_second = k_base + t_offset * stride_kt + (offs_k - K_HALF) * stride_kk

            k_rot_first = tl.load(k_rot_ptrs_first, mask=mask_first_half & ((offs_k + K_HALF) < K), other=0.0).to(tl.float32)
            k_rot_second = tl.load(k_rot_ptrs_second, mask=mask_second_half & ((offs_k - K_HALF) >= 0), other=0.0).to(tl.float32)

            k_rot_half = tl.where(mask_first_half, -k_rot_first, k_rot_second)

            # k_embed = k * cos + rotate_half(k) * sin
            k_embed = k_vals * cos_vals + k_rot_half * sin_vals

            # Update max for scale
            k_abs = tl.abs(k_embed)
            k_max = tl.maximum(k_max, k_abs)

            # Store rotated k
            for k_idx in range(K):
                if k_idx < K:
                    k_rotated = tl.where((t_idx == tl.arange(0, BS)[:, None]) & (k_idx == offs_k[None, :]),
                                        k_embed[None, :],
                                        k_rotated)

    # Compute scale
    k_scale = k_max / QZERO
    k_scale = tl.maximum(k_scale, EPS)

    # Quantize and store results
    for t_idx in range(BS):
        t_offset = t_start + t_idx
        if t_offset < T:
            # Get rotated k for this token
            k_vals = tl.zeros([K], dtype=tl.float32)
            for k_idx in range(K):
                k_vals = tl.where(offs_k == k_idx,
                                 tl.sum(tl.where(tl.arange(0, BS) == t_idx, k_rotated[:, k_idx], 0.0)),
                                 k_vals)

            # Quantize
            k_normalized = k_vals / k_scale
            k_q_float = tl.maximum(tl.minimum((k_normalized + QZERO), QMAX), 0.0)
            k_q_int = k_q_float.to(tl.int32)

            # Dequantize for residual
            k_dequant = (k_q_float - QZERO) * k_scale
            k_res = k_vals - k_dequant

            # Store residual
            res_ptrs = (k_residual_ptr + pid_b * stride_rb + t_offset * stride_rt +
                       pid_h * stride_rh + offs_k * stride_rk)
            tl.store(res_ptrs, k_res.to(tl.float16), mask=offs_k < K)

            # Pack and store quantized values
            if K_BITS == 2:
                for pack_idx in range(K_PACKED):
                    k_base_idx = pack_idx * VALS_PER_BYTE
                    packed = 0
                    for i in range(VALS_PER_BYTE):
                        k_idx = k_base_idx + i
                        if k_idx < K:
                            val = tl.sum(tl.where(offs_k == k_idx, k_q_int, 0))
                            packed = packed | (val << (i * 2))

                    q_ptr = (k_q_ptr + pid_b * stride_qb + t_offset * stride_qt +
                            pid_h * stride_qh + pack_idx * stride_qk)
                    tl.store(q_ptr, packed.to(tl.uint8))
            else:  # K_BITS == 4
                for pack_idx in range(K_PACKED):
                    k_base_idx = pack_idx * VALS_PER_BYTE
                    packed = 0
                    for i in range(VALS_PER_BYTE):
                        k_idx = k_base_idx + i
                        if k_idx < K:
                            val = tl.sum(tl.where(offs_k == k_idx, k_q_int, 0))
                            packed = packed | (val << (i * 4))

                    q_ptr = (k_q_ptr + pid_b * stride_qb + t_offset * stride_qt +
                            pid_h * stride_qh + pack_idx * stride_qk)
                    tl.store(q_ptr, packed.to(tl.uint8))

    # Store scale
    scale_ptrs = (k_scale_ptr + pid_b * stride_sb + pid_block * stride_sblk +
                 pid_h * stride_sh + offs_k * stride_sk)
    tl.store(scale_ptrs, k_scale, mask=offs_k < K)


def fused_rope_and_quantize(
    k: torch.Tensor,  # [B, T, HKV, K]
    cos: torch.Tensor,  # [B, T, K] or [1, T, K]
    sin: torch.Tensor,  # [B, T, K] or [1, T, K]
    block_size: int = 128,
    k_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused RoPE + Quantization

    Args:
        k: Key states [B, T, HKV, K]
        cos: Cosine for RoPE [B, T, K] or [1, T, K]
        sin: Sine for RoPE [B, T, K] or [1, T, K]
        block_size: Block size for quantization
        k_bits: Quantization bits (2 or 4)

    Returns:
        k_q: Quantized keys [B, T, HKV, K_packed]
        k_scale: Scales [B, num_blocks, HKV, K]
        k_residual: Residuals [B, T, HKV, K]
    """
    B, T, HKV, K = k.shape

    # Ensure T is divisible by block_size
    assert T % block_size == 0, f"T={T} must be divisible by block_size={block_size}"
    assert K % 2 == 0, f"K={K} must be even for RoPE"

    num_blocks = T // block_size
    VALS_PER_BYTE = 8 // k_bits
    K_PACKED = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    # Allocate output tensors
    k_q = torch.empty((B, T, HKV, K_PACKED), dtype=torch.uint8, device=k.device)
    k_scale = torch.empty((B, num_blocks, HKV, K), dtype=torch.float32, device=k.device)
    k_residual = torch.empty((B, T, HKV, K), dtype=torch.float16, device=k.device)

    # Launch kernel
    grid = (num_blocks, HKV, B)

    fused_rope_quantize_kernel[grid](
        k, cos, sin,
        k_q, k_scale, k_residual,
        B, T, HKV, K, K_PACKED, block_size, k_bits,
        # Strides for k
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for cos
        cos.stride(0), cos.stride(1), cos.stride(2),
        # Strides for sin
        sin.stride(0), sin.stride(1), sin.stride(2),
        # Strides for k_q
        k_q.stride(0), k_q.stride(1), k_q.stride(2), k_q.stride(3),
        # Strides for k_scale
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        # Strides for k_residual
        k_residual.stride(0), k_residual.stride(1), k_residual.stride(2), k_residual.stride(3),
    )

    return k_q, k_scale, k_residual


# ============================================================================
# Reference implementation for correctness testing
# ============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ref(k, cos, sin):
    """Reference RoPE implementation"""
    # cos, sin: [B, T, K]
    # k: [B, T, HKV, K]
    cos = cos.unsqueeze(2)  # [B, T, 1, K]
    sin = sin.unsqueeze(2)  # [B, T, 1, K]
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


def quantize_symmetric_blocks_ref(
    k_blocks: torch.Tensor,
    block_size: int,
    k_bits: int = 2,
    eps: float = 1e-8,
):
    """Reference quantization implementation"""
    B, T, HKV, K = k_blocks.shape
    assert T % block_size == 0

    dtype = k_blocks.dtype
    num_blocks = T // block_size

    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE = 8 // k_bits
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    k_blocks = k_blocks.reshape(B, num_blocks, block_size, HKV, K)

    # Compute scale per block
    k_abs_max = k_blocks.abs().amax(dim=2)  # [B, num_blocks, HKV, K]
    k_scale = (k_abs_max / QZERO).clamp(min=eps)

    # Quantize
    k_norm = k_blocks / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_packed, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 4
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_packed)

    # Dequantize for residual
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_blocks - k_dequant
    k_residual = k_residual.reshape(B, T, HKV, K).to(torch.float16)

    return k_q_packed, k_scale, k_residual


def reference_rope_and_quantize(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_size: int = 128,
    k_bits: int = 2,
):
    """Reference implementation: RoPE then quantize"""
    # Apply RoPE
    k_rotated = apply_rotary_pos_emb_ref(k, cos, sin)

    # Quantize
    k_q, k_scale, k_residual = quantize_symmetric_blocks_ref(
        k_rotated, block_size, k_bits
    )

    return k_q, k_scale, k_residual


# ============================================================================
# Testing
# ============================================================================

def test_fused_rope_quantize():
    """Test correctness of fused kernel"""
    print("Testing fused RoPE + Quantization kernel...")

    # Test configuration
    B, T, HKV, K = 1, 256, 8, 128
    block_size = 128
    k_bits = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available, skipping test")
        return

    # Generate test data
    torch.manual_seed(42)
    k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

    print(f"Input shape: k={k.shape}, cos={cos.shape}, sin={sin.shape}")
    print(f"Block size: {block_size}, k_bits: {k_bits}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    k_q_ref, k_scale_ref, k_res_ref = reference_rope_and_quantize(
        k, cos, sin, block_size, k_bits
    )
    print(f"Reference output shapes: k_q={k_q_ref.shape}, k_scale={k_scale_ref.shape}, k_res={k_res_ref.shape}")

    # Run fused kernel
    print("Running fused kernel...")
    try:
        k_q_fused, k_scale_fused, k_res_fused = fused_rope_and_quantize(
            k, cos, sin, block_size, k_bits
        )
        print(f"Fused output shapes: k_q={k_q_fused.shape}, k_scale={k_scale_fused.shape}, k_res={k_res_fused.shape}")
    except Exception as e:
        print(f"Fused kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare results
    print("\nComparing results...")

    # Compare quantized values
    q_match = torch.all(k_q_ref == k_q_fused).item()
    print(f"Quantized values match: {q_match}")
    if not q_match:
        diff = (k_q_ref != k_q_fused).sum().item()
        print(f"  Differences: {diff} / {k_q_ref.numel()} elements ({100*diff/k_q_ref.numel():.2f}%)")
        # Show some examples
        diff_mask = k_q_ref != k_q_fused
        if diff_mask.any():
            idx = torch.where(diff_mask)
            print(f"  Example diff at {tuple(i[0].item() for i in idx)}: ref={k_q_ref[idx][0].item()}, fused={k_q_fused[idx][0].item()}")

    # Compare scales (convert to same dtype)
    k_scale_ref_f32 = k_scale_ref.float()
    k_scale_fused_f32 = k_scale_fused.float()
    scale_diff = (k_scale_ref_f32 - k_scale_fused_f32).abs()
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    print(f"Scale max diff: {scale_max_diff:.6f}, mean diff: {scale_mean_diff:.6f}")

    # Compare residuals
    res_diff = (k_res_ref - k_res_fused).abs()
    res_max_diff = res_diff.max().item()
    res_mean_diff = res_diff.mean().item()
    print(f"Residual max diff: {res_max_diff:.6f}, mean diff: {res_mean_diff:.6f}")

    # Check if results are close enough
    scale_close = torch.allclose(k_scale_ref_f32, k_scale_fused_f32, rtol=1e-3, atol=1e-4)
    res_close = torch.allclose(k_res_ref, k_res_fused, rtol=1e-2, atol=0.1)

    print(f"\nScales close: {scale_close}")
    print(f"Residuals close: {res_close}")

    if q_match and scale_close and res_close:
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        return False


if __name__ == "__main__":
    success = test_fused_rope_quantize()
    exit(0 if success else 1)

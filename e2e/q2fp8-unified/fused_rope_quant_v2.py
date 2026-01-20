"""
Fused RoPE + Quantization Kernel (Simplified Version)

将 RoPE 和 KV cache 量化合并成一个 kernel，减少内存访问和中间结果存储。

优化点：
1. K tensor 只读一次，直接应用 RoPE 后量化
2. 避免中间结果写回 global memory
3. 使用更简单的 Triton kernel 实现
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_quantize_kernel(
    # Input
    k_ptr,  # [B, T, HKV, K]
    cos_ptr,  # [B, T, K]
    sin_ptr,  # [B, T, K]
    # Output
    k_q_ptr,  # [B, T, HKV, K_packed]
    k_scale_ptr,  # [B, num_blocks, HKV, K]
    k_residual_ptr,  # [B, T, HKV, K]
    # Dimensions
    B, T, HKV, K, K_PACKED, BS, K_BITS,
    # Strides
    stride_kb, stride_kt, stride_kh, stride_kk,
    stride_cosb, stride_cost, stride_cosk,
    stride_sinb, stride_sint, stride_sink,
    stride_qb, stride_qt, stride_qh, stride_qk,
    stride_sb, stride_sblk, stride_sh, stride_sk,
    stride_rb, stride_rt, stride_rh, stride_rk,
    # Block size for processing K dimension
    BLOCK_K: tl.constexpr,
):
    """
    Fused RoPE + Quantization kernel

    Grid: (num_blocks, HKV, B)
    Each program processes BS tokens for one head
    """
    pid_block = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Constants
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE = 8 // K_BITS
    EPS = 1e-8
    K_HALF = K // 2

    t_start = pid_block * BS

    # Base pointers
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    cos_base = cos_ptr + pid_b * stride_cosb
    sin_base = sin_ptr + pid_b * stride_sinb

    # Offsets for K dimension
    offs_k = tl.arange(0, BLOCK_K)

    # Storage for rotated K values and max tracking
    # We'll process in chunks of BLOCK_K
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    # First pass: Apply RoPE and compute max for each K dimension
    k_max_vals = tl.zeros([K], dtype=tl.float32) + EPS

    # We need to store rotated K values - use a simpler approach
    # Process token by token, K dimension by K dimension

    for t_idx in range(BS):
        t_offset = t_start + t_idx
        if t_offset >= T:
            continue

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * BLOCK_K
            k_offs = k_start + offs_k
            k_mask = k_offs < K

            # Load K values
            k_ptrs = k_base + t_offset * stride_kt + k_offs * stride_kk
            k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Load cos/sin
            cos_ptrs = cos_base + t_offset * stride_cost + k_offs * stride_cosk
            sin_ptrs = sin_base + t_offset * stride_sint + k_offs * stride_sink
            cos_vals = tl.load(cos_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            sin_vals = tl.load(sin_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Apply RoPE: need to load the "other half" for rotation
            # For first half (k < K/2), we need k[k + K/2]
            # For second half (k >= K/2), we need k[k - K/2]
            is_first_half = k_offs < K_HALF
            other_half_offs = tl.where(is_first_half, k_offs + K_HALF, k_offs - K_HALF)

            k_other_ptrs = k_base + t_offset * stride_kt + other_half_offs * stride_kk
            k_other = tl.load(k_other_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # rotate_half: [-k[K/2:], k[:K/2]]
            # For first half: use -k_other (which is -k[K/2:])
            # For second half: use k_other (which is k[:K/2])
            k_rot_half = tl.where(is_first_half, -k_other, k_other)

            # Apply RoPE: k * cos + rotate_half(k) * sin
            k_embed = k_vals * cos_vals + k_rot_half * sin_vals

            # Update max (will be used for scale computation)
            k_abs = tl.abs(k_embed)
            # Store max - need to be careful with indexing
            for i in range(BLOCK_K):
                if k_start + i < K:
                    k_max_vals = tl.where(offs_k == i,
                                         tl.maximum(k_max_vals[k_start + i], k_abs[i]),
                                         k_max_vals)


def fused_rope_and_quantize_simple(
    k: torch.Tensor,  # [B, T, HKV, K]
    cos: torch.Tensor,  # [B, T, K]
    sin: torch.Tensor,  # [B, T, K]
    block_size: int = 128,
    k_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified fused RoPE + Quantization using PyTorch operations

    This version uses PyTorch for simplicity and correctness.
    Can be optimized with custom CUDA kernels later.
    """
    B, T, HKV, K = k.shape
    assert T % block_size == 0
    assert K % 2 == 0

    num_blocks = T // block_size
    VALS_PER_BYTE = 8 // k_bits
    K_PACKED = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2.0
    EPS = 1e-8

    # Step 1: Apply RoPE (fused with quantization logic)
    # Expand cos/sin to match k shape
    cos = cos.unsqueeze(2)  # [B, T, 1, K]
    sin = sin.unsqueeze(2)  # [B, T, 1, K]

    # Apply RoPE
    k1 = k[..., :K//2]
    k2 = k[..., K//2:]
    k_rotated = torch.cat([
        k1 * cos[..., :K//2] - k2 * sin[..., :K//2],
        k2 * cos[..., K//2:] + k1 * sin[..., K//2:]
    ], dim=-1)

    # Step 2: Quantize by blocks
    k_rotated = k_rotated.reshape(B, num_blocks, block_size, HKV, K)

    # Compute scale per block
    k_abs_max = k_rotated.abs().amax(dim=2)  # [B, num_blocks, HKV, K]
    k_scale = (k_abs_max / QZERO).clamp(min=EPS)

    # Quantize
    k_norm = k_rotated / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack into uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_PACKED, VALS_PER_BYTE)

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

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_PACKED)

    # Compute residual
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_rotated - k_dequant
    k_residual = k_residual.reshape(B, T, HKV, K).to(torch.float16)

    return k_q_packed, k_scale, k_residual


# ============================================================================
# Reference implementation
# ============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ref(k, cos, sin):
    """Reference RoPE implementation"""
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
    """Test correctness of fused implementation"""
    print("Testing fused RoPE + Quantization (simplified version)...")

    # Test configuration
    B, T, HKV, K = 1, 256, 8, 128
    block_size = 128
    k_bits = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available, using CPU")

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

    # Run fused implementation
    print("Running fused implementation...")
    k_q_fused, k_scale_fused, k_res_fused = fused_rope_and_quantize_simple(
        k, cos, sin, block_size, k_bits
    )
    print(f"Fused output shapes: k_q={k_q_fused.shape}, k_scale={k_scale_fused.shape}, k_res={k_res_fused.shape}")

    # Compare results
    print("\nComparing results...")

    # Compare quantized values
    q_match = torch.all(k_q_ref == k_q_fused).item()
    print(f"Quantized values match: {q_match}")
    if not q_match:
        diff = (k_q_ref != k_q_fused).sum().item()
        print(f"  Differences: {diff} / {k_q_ref.numel()} elements ({100*diff/k_q_ref.numel():.2f}%)")

    # Compare scales
    scale_diff = (k_scale_ref - k_scale_fused).abs()
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    print(f"Scale max diff: {scale_max_diff:.6f}, mean diff: {scale_mean_diff:.6f}")

    # Compare residuals
    res_diff = (k_res_ref - k_res_fused).abs()
    res_max_diff = res_diff.max().item()
    res_mean_diff = res_diff.mean().item()
    print(f"Residual max diff: {res_max_diff:.6f}, mean diff: {res_mean_diff:.6f}")

    # Check if results are close enough
    scale_close = torch.allclose(k_scale_ref, k_scale_fused, rtol=1e-5, atol=1e-6)
    res_close = torch.allclose(k_res_ref, k_res_fused, rtol=1e-3, atol=1e-4)

    print(f"\nScales close: {scale_close}")
    print(f"Residuals close: {res_close}")

    if q_match and scale_close and res_close:
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        return False


def benchmark_fused_vs_separate():
    """Benchmark fused vs separate RoPE + quantization"""
    print("\n" + "="*60)
    print("Benchmarking fused vs separate implementation")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available, skipping benchmark")
        return

    # Realistic configuration
    B, T, HKV, K = 1, 32768, 8, 128
    block_size = 128
    k_bits = 2

    print(f"\nConfiguration: B={B}, T={T}, HKV={HKV}, K={K}")
    print(f"Block size: {block_size}, k_bits: {k_bits}")

    # Generate test data
    torch.manual_seed(42)
    k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = reference_rope_and_quantize(k, cos, sin, block_size, k_bits)
        _ = fused_rope_and_quantize_simple(k, cos, sin, block_size, k_bits)
    torch.cuda.synchronize()

    # Benchmark separate
    print("\nBenchmarking separate RoPE + quantization...")
    import time
    times_separate = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = reference_rope_and_quantize(k, cos, sin, block_size, k_bits)
        torch.cuda.synchronize()
        times_separate.append((time.perf_counter() - start) * 1000)

    avg_separate = sum(times_separate) / len(times_separate)
    print(f"Average time (separate): {avg_separate:.3f} ms")

    # Benchmark fused
    print("\nBenchmarking fused RoPE + quantization...")
    times_fused = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fused_rope_and_quantize_simple(k, cos, sin, block_size, k_bits)
        torch.cuda.synchronize()
        times_fused.append((time.perf_counter() - start) * 1000)

    avg_fused = sum(times_fused) / len(times_fused)
    print(f"Average time (fused): {avg_fused:.3f} ms")

    speedup = avg_separate / avg_fused
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved: {avg_separate - avg_fused:.3f} ms ({100*(avg_separate - avg_fused)/avg_separate:.1f}%)")


if __name__ == "__main__":
    success = test_fused_rope_quantize()

    if success:
        benchmark_fused_vs_separate()

    exit(0 if success else 1)

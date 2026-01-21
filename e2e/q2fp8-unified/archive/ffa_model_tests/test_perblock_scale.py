#!/usr/bin/env python
"""
Test: Verify per-block scale implementation in the kernel.
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')

import torch
import math

# Import the kernel function
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold-opt/attn_kernel')
from attn_q2fp8_sym_mask import attn_forward_decode_quantized

def create_test_data(B, T, HKV, HQ, K, V, BS=128, k_bits=2):
    """Create test data for the kernel."""
    torch.manual_seed(42)

    # Q: [B, 1, HQ, K]
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device='cuda')

    # K: [B, T, HKV, K] - original keys
    k_orig = torch.randn(B, T, HKV, K, dtype=torch.float16, device='cuda')

    # V: [B, T, HKV, V]
    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device='cuda')

    # Quantize K per-block
    NTB = (T + BS - 1) // BS
    VALS_PER_BYTE = 8 // k_bits
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2

    k_q = torch.zeros(B, T, HKV, K_packed, dtype=torch.uint8, device='cuda')
    k_scale = torch.zeros(B, NTB, HKV, K, dtype=torch.float32, device='cuda')
    k_residual = torch.zeros(B, T, HKV, K, dtype=torch.float16, device='cuda')

    for b in range(NTB):
        start = b * BS
        end = min(start + BS, T)
        block_len = end - start

        k_block = k_orig[:, start:end, :, :]  # [B, block_len, HKV, K]

        # Compute per-block scale: abs_max per channel
        abs_max = k_block.abs().amax(dim=1, keepdim=True)  # [B, 1, HKV, K]
        scale = abs_max / QZERO  # [B, 1, HKV, K]
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        k_scale[:, b, :, :] = scale.squeeze(1)  # [B, HKV, K]

        # Quantize
        k_scaled = k_block / scale  # [B, block_len, HKV, K]
        k_int = torch.clamp(torch.round(k_scaled + QZERO), 0, QMAX).to(torch.int32)

        # Pack 2-bit values into uint8
        for i in range(VALS_PER_BYTE):
            if i * K_packed < K:
                end_k = min((i + 1) * K_packed, K)
                shift = i * k_bits
                k_q[:, start:end, :, :] |= (k_int[..., i::VALS_PER_BYTE].to(torch.uint8) << shift)

        # Compute residual
        k_dequant = (k_int.float() - QZERO) * scale
        k_residual[:, start:end, :, :] = (k_block - k_dequant).to(torch.float16)

    return q, k_q, k_scale, v, k_residual, k_orig


def test_perblock_scale():
    print("=" * 70)
    print("Testing Per-Block Scale Implementation")
    print("=" * 70)

    # Test parameters
    B = 1
    T = 256  # 2 blocks with BS=128
    HKV = 8
    HQ = 32  # GQA with G=4
    K = 128
    V = 128
    BS = 128
    k_bits = 2

    print(f"\nTest config: B={B}, T={T}, HKV={HKV}, HQ={HQ}, K={K}, V={V}, BS={BS}")

    # Create test data
    print("\n[1] Creating test data with per-block quantization...")
    q, k_q, k_scale_perblock, v, k_residual, k_orig = create_test_data(
        B, T, HKV, HQ, K, V, BS, k_bits
    )

    print(f"  q shape: {q.shape}")
    print(f"  k_q shape: {k_q.shape}")
    print(f"  k_scale (per-block) shape: {k_scale_perblock.shape}")
    print(f"  v shape: {v.shape}")
    print(f"  k_residual shape: {k_residual.shape}")

    # Test 1: Per-block scale
    print("\n[2] Testing with per-block scale [B, NTB, HKV, K]...")
    try:
        output_perblock = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale_perblock,  # [B, NTB, HKV, K]
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            BS=BS,
            delta=5.0,
            use_fp8_residual=True,
        )
        print(f"  Output shape: {output_perblock.shape}")
        print(f"  Output range: [{output_perblock.min():.4f}, {output_perblock.max():.4f}]")
        print("  SUCCESS: Per-block scale works!")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Global scale (amax reduction - old behavior)
    print("\n[3] Testing with global scale [B, HKV, K] (amax reduction)...")
    k_scale_global = k_scale_perblock.amax(dim=1)  # [B, HKV, K]
    print(f"  k_scale (global) shape: {k_scale_global.shape}")

    try:
        output_global = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale_global,  # [B, HKV, K]
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            BS=BS,
            delta=5.0,
            use_fp8_residual=True,
        )
        print(f"  Output shape: {output_global.shape}")
        print(f"  Output range: [{output_global.min():.4f}, {output_global.max():.4f}]")
        print("  SUCCESS: Global scale also works!")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare outputs
    print("\n[4] Comparing outputs...")
    diff = (output_perblock - output_global).abs()
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff: {diff.max():.6f}")

    # Per-block scale should be more accurate (diff should be non-zero when scales differ)
    scale_diff = (k_scale_perblock[:, 0, :, :] - k_scale_perblock[:, 1, :, :]).abs().mean()
    print(f"  Scale diff between blocks: {scale_diff:.6f}")

    if scale_diff > 0.01:
        print("\n  Blocks have different scales, per-block should be more accurate.")
        if diff.mean() > 1e-6:
            print("  Outputs differ as expected (per-block uses exact scale per block).")

    # Test 3: return_lse for accurate merging
    print("\n[5] Testing return_lse feature...")
    try:
        output, m, l = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale_perblock,
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            BS=BS,
            delta=5.0,
            use_fp8_residual=True,
            return_lse=True,
        )
        print(f"  Output shape: {output.shape}")
        print(f"  m (max) shape: {m.shape}, range: [{m.min():.4f}, {m.max():.4f}]")
        print(f"  l (sum) shape: {l.shape}, range: [{l.min():.4f}, {l.max():.4f}]")
        print("  SUCCESS: return_lse works!")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_perblock_scale()

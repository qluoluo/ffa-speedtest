#!/usr/bin/env python
"""
Test edge cases in the kernel.
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold-opt/attn_kernel')

import torch
from attn_q2fp8_sym_mask import attn_forward_decode_quantized


def test_all_blocks_pruned():
    """Test behavior when delta is very high, causing all blocks to be pruned."""
    print("=" * 60)
    print("Test: All blocks pruned (very high delta)")
    print("=" * 60)

    B, T, HKV, HQ, K, V = 1, 256, 8, 32, 128, 128
    BS = 128
    k_bits = 2
    VALS_PER_BYTE = 4
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    torch.manual_seed(42)
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device='cuda')
    k_q = torch.randint(0, 256, (B, T, HKV, K_packed), dtype=torch.uint8, device='cuda')
    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device='cuda')
    k_scale = torch.ones(B, HKV, K, dtype=torch.float32, device='cuda') * 0.1
    k_residual = torch.zeros(B, T, HKV, K, dtype=torch.float16, device='cuda')

    # Very high delta should prune all blocks
    delta = 1000.0

    print(f"\nUsing delta={delta} (very high, should prune all blocks)")

    try:
        output, m, l = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            BS=BS,
            delta=delta,
            use_fp8_residual=True,
            return_lse=True,
        )
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"m (max) range: [{m.min():.4f}, {m.max():.4f}]")
        print(f"l (sum) range: [{l.min():.4f}, {l.max():.4f}]")

        # Check if l has zeros (all blocks pruned)
        l_zeros = (l == 0).sum().item()
        if l_zeros > 0:
            print(f"\nWARNING: {l_zeros} heads have l=0 (all blocks pruned)!")
            print("This may cause division by zero in merge!")

            # Check if output has NaN/Inf
            if torch.isnan(output).any():
                print("ERROR: Output contains NaN!")
            elif torch.isinf(output).any():
                print("ERROR: Output contains Inf!")
            else:
                print("Output is finite (kernel handles l=0 gracefully)")
        else:
            print("\nNo heads with l=0, some blocks were kept.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_single_block():
    """Test with T < BS (no full blocks to quantize)."""
    print("\n" + "=" * 60)
    print("Test: Single partial block (T < BS)")
    print("=" * 60)

    B, T, HKV, HQ, K, V = 1, 64, 8, 32, 128, 128  # T < BS=128
    BS = 128
    k_bits = 2
    VALS_PER_BYTE = 4
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    torch.manual_seed(42)
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device='cuda')
    k_q = torch.randint(0, 256, (B, T, HKV, K_packed), dtype=torch.uint8, device='cuda')
    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device='cuda')
    k_scale = torch.ones(B, HKV, K, dtype=torch.float32, device='cuda') * 0.1
    k_residual = torch.zeros(B, T, HKV, K, dtype=torch.float16, device='cuda')

    print(f"\nT={T}, BS={BS}, NTB={max(1, (T + BS - 1) // BS)}")

    try:
        output = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            BS=BS,
            delta=5.0,
            use_fp8_residual=True,
        )
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        if torch.isnan(output).any():
            print("ERROR: Output contains NaN!")
        elif torch.isinf(output).any():
            print("ERROR: Output contains Inf!")
        else:
            print("SUCCESS: Single block test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_threshold_sampling():
    """Test if threshold sampling is representative."""
    print("\n" + "=" * 60)
    print("Test: Threshold sampling representativeness")
    print("=" * 60)

    # Create data where important attention is at the END of each block
    B, T, HKV, HQ, K, V = 1, 256, 8, 32, 128, 128  # 2 blocks
    BS = 128
    k_bits = 2
    VALS_PER_BYTE = 4
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    QMAX = 3
    QZERO = 1.5

    torch.manual_seed(42)
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device='cuda')

    # Create k_q where the last tokens in each block have high values
    # but first T_BS=16 tokens have low values
    k_q = torch.zeros(B, T, HKV, K_packed, dtype=torch.uint8, device='cuda')

    # First 16 tokens of each block: low values (q=0 or 1)
    # Last tokens: high values (q=2 or 3)
    for block in range(2):
        start = block * BS
        # First 16 tokens: pack zeros
        k_q[:, start:start+16, :, :] = 0  # All q=0
        # Rest of block: pack max values (0xFF = all 3s for 2-bit)
        k_q[:, start+16:start+BS, :, :] = 0xFF

    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device='cuda')
    k_scale = torch.ones(B, HKV, K, dtype=torch.float32, device='cuda') * 0.5
    k_residual = torch.zeros(B, T, HKV, K, dtype=torch.float16, device='cuda')

    print("\nData setup:")
    print("  - First 16 tokens of each block: k_q = 0 (low attention)")
    print("  - Rest of each block: k_q = max (high attention)")
    print("  - Threshold kernel only samples first 16 tokens!")

    # Test with low delta (should NOT prune blocks)
    delta = 2.0
    print(f"\nUsing delta={delta}")

    output, skip_ratio = attn_forward_decode_quantized(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        v=v,
        k_residual=k_residual,
        k_bits=k_bits,
        BS=BS,
        delta=delta,
        use_fp8_residual=True,
        return_skip_ratio=True,
    )

    print(f"Skip ratio: {skip_ratio:.2%}")
    if skip_ratio > 0.5:
        print("WARNING: High skip ratio suggests threshold sampling may be unrepresentative!")
        print("         The kernel only samples first T_BS=16 tokens for threshold.")
    else:
        print("Skip ratio is reasonable.")


if __name__ == "__main__":
    test_all_blocks_pruned()
    test_single_block()
    test_threshold_sampling()
    print("\n" + "=" * 60)
    print("Edge case tests completed!")
    print("=" * 60)

#!/usr/bin/env python3
"""
Test script for unified Q2FP8 kernel with FP16 current tokens support
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "attn_kernel"))

import torch
import time
from attn_q2fp8_unified import attn_forward_decode_quantized


def quantize_symmetric_2bit(k: torch.Tensor):
    """Simple 2-bit symmetric quantization"""
    B, T, HKV, K = k.shape

    # Compute per-token scale
    k_abs_max = k.abs().amax(dim=-1, keepdim=True)  # [B, T, HKV, 1]
    k_scale = k_abs_max / 1.5  # QZERO = 1.5 for 2-bit
    k_scale = k_scale.clamp(min=1e-8)

    # Quantize
    k_norm = k / k_scale
    k_q_float = (k_norm + 1.5).round().clamp(0, 3)

    # Pack to uint8
    K_packed = (K + 3) // 4
    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, T, HKV, K_packed, 4)
    k_q_packed = (
        k_q_int[..., 0] |
        (k_q_int[..., 1] << 2) |
        (k_q_int[..., 2] << 4) |
        (k_q_int[..., 3] << 6)
    ).to(torch.uint8)

    # Compute residual
    k_dequant = (k_q_float - 1.5) * k_scale
    k_residual = k - k_dequant

    # Convert scale to per-block: [B, HKV, K]
    # Use global scale (max across all tokens)
    k_scale_global = k_abs_max.amax(dim=1) / 1.5  # [B, HKV, 1]
    k_scale_global = k_scale_global.clamp(min=1e-8)
    k_scale_block = k_scale_global.expand(B, HKV, K)  # [B, HKV, K]

    return k_q_packed, k_scale_block, k_residual


def test_basic_functionality():
    """Test 1: Basic functionality without current tokens"""
    print("=" * 70)
    print("Test 1: Basic functionality (no current tokens)")
    print("=" * 70)

    device = torch.device("cuda:0")
    dtype = torch.float16

    B, T, HQ, HKV, K, V = 1, 256, 32, 8, 128, 128

    # Create inputs
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Quantize K
    k_q, k_scale, k_residual = quantize_symmetric_2bit(k)

    print(f"Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k_q: {k_q.shape}")
    print(f"  k_scale: {k_scale.shape}")
    print(f"  v: {v.shape}")

    try:
        # Run kernel without current tokens
        output = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_current=None,
            v_current=None,
            current_len=0,
            k_residual=k_residual,
            k_bits=2,
            BS=128,
            delta=5.0,
            use_fp8_residual=True,
        )

        print(f"\nOutput shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("✅ Test 1 PASSED")
        return True
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_current_tokens():
    """Test 2: With FP16 current tokens"""
    print("\n" + "=" * 70)
    print("Test 2: With FP16 current tokens")
    print("=" * 70)

    device = torch.device("cuda:0")
    dtype = torch.float16

    B, T, HQ, HKV, K, V = 1, 256, 32, 8, 128, 128
    current_len = 64  # 64 tokens in current buffer
    max_current = 128

    # Create inputs
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Quantize K
    k_q, k_scale, k_residual = quantize_symmetric_2bit(k)

    # Create current buffers
    k_current = torch.randn(B, max_current, HKV, K, device=device, dtype=dtype)
    v_current = torch.randn(B, max_current, HKV, V, device=device, dtype=dtype)

    print(f"Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k_q: {k_q.shape}")
    print(f"  k_scale: {k_scale.shape}")
    print(f"  v: {v.shape}")
    print(f"  k_current: {k_current.shape}")
    print(f"  v_current: {v_current.shape}")
    print(f"  current_len: {current_len}")

    try:
        # Run kernel with current tokens
        output, skip_ratio = attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_current=k_current,
            v_current=v_current,
            current_len=current_len,
            k_residual=k_residual,
            k_bits=2,
            BS=128,
            delta=5.0,
            use_fp8_residual=True,
            return_skip_ratio=True,
            max_current=max_current,
        )

        print(f"\nOutput shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Skip ratio: {skip_ratio:.4f}")
        print("✅ Test 2 PASSED")
        return True
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test 3: Performance comparison"""
    print("\n" + "=" * 70)
    print("Test 3: Performance comparison")
    print("=" * 70)

    device = torch.device("cuda:0")
    dtype = torch.float16

    B, T, HQ, HKV, K, V = 1, 256*1024, 32, 8, 128, 128  # 256K sequence
    current_len = 64
    max_current = 128

    # Create inputs
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Quantize K
    k_q, k_scale, k_residual = quantize_symmetric_2bit(k)

    # Create current buffers
    k_current = torch.randn(B, max_current, HKV, K, device=device, dtype=dtype)
    v_current = torch.randn(B, max_current, HKV, V, device=device, dtype=dtype)

    print(f"Sequence length: {T}")
    print(f"Current length: {current_len}")

    # Warmup
    for _ in range(10):
        _ = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, v=v,
            k_current=k_current, v_current=v_current, current_len=current_len,
            k_residual=k_residual, k_bits=2, BS=128, delta=5.0,
            use_fp8_residual=True, max_current=max_current,
        )
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, v=v,
            k_current=k_current, v_current=v_current, current_len=current_len,
            k_residual=k_residual, k_bits=2, BS=128, delta=5.0,
            use_fp8_residual=True, max_current=max_current,
        )
    end.record()
    end.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters
    print(f"\nAverage time: {elapsed_ms:.4f} ms")
    print("✅ Test 3 PASSED")
    return True


def main():
    print("Testing Unified Q2FP8 Kernel")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Basic functionality", test_basic_functionality()))
    results.append(("With current tokens", test_with_current_tokens()))

    try:
        results.append(("Performance", test_performance()))
    except Exception as e:
        print(f"Performance test skipped: {e}")
        results.append(("Performance", False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

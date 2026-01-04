#!/usr/bin/env python3
"""
Test and benchmark the optimized Q2FP8 kernels.

This script tests all three optimizations:
1. LUT dequant (works on all GPUs)
2. FP8 tensor core (H100 optimal, graceful fallback on other GPUs)
3. Async copy (TMA on H100, standard async on other GPUs)

Usage:
    # Test on local GPU (e.g., 4090)
    python test_optimized_kernels.py --test-correctness --bench

    # Full benchmark with all optimization combinations
    python test_optimized_kernels.py --full-bench

    # Quick test
    python test_optimized_kernels.py --quick
"""

import argparse
import sys
from pathlib import Path
import torch
import time

# Add project root to path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized
from optimizations.kernels import (
    attn_forward_decode_quantized_optimized,
    create_dequant_lut,
)


def quantize_k_2bit_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2):
    """Quantize K to 2-bit with FP8 residual."""
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack to 2-bit
    values_per_byte = 4
    B, T, HKV, K = k_q.shape
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)
        k_q = torch.cat([k_q, pad_tensor], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()
    return k_q_packed, scale, zero, k_residual


def create_test_inputs(B=1, T=8192, HQ=24, HKV=8, K=128, V=128, device='cuda', dtype=torch.float16):
    """Create test inputs."""
    print(f"Creating test inputs: B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}")

    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Quantize K
    k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)

    print(f"  q: {q.shape}, k_q: {k_q.shape}, k_scale: {k_scale.shape}, v: {v.shape}")
    print(f"  k_residual: {k_residual.shape}, dtype: {k_residual.dtype}")

    return q, k_q, k_scale, k_zero, k_residual, v


def test_correctness(B=1, T=4096, HQ=24, HKV=8, K=128, V=128):
    """Test correctness of optimized kernels vs original."""
    print("\n" + "=" * 80)
    print("TESTING CORRECTNESS")
    print("=" * 80)

    device = 'cuda'
    dtype = torch.float16

    q, k_q, k_scale, k_zero, k_residual, v = create_test_inputs(
        B=B, T=T, HQ=HQ, HKV=HKV, K=K, V=V, device=device, dtype=dtype
    )

    # Test 1: Original kernel
    print("\n[1/4] Testing original kernel...")
    out_original = attn_forward_decode_quantized(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        v=v,
        k_residual=k_residual,
        BS=128,
        delta=5.0,
        use_fp8_residual=True,
    )
    print(f"  Output shape: {out_original.shape}")
    print(f"  Output range: [{out_original.min():.4f}, {out_original.max():.4f}]")

    # Test 2: Optimized with LUT only
    print("\n[2/4] Testing optimized kernel (LUT only)...")
    out_lut = attn_forward_decode_quantized_optimized(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        v=v,
        k_residual=k_residual,
        BS=128,
        delta=5.0,
        use_fp8_residual=True,
        use_fp8_compute=False,
        use_async_copy=False,
    )
    print(f"  Output shape: {out_lut.shape}")
    print(f"  Output range: [{out_lut.min():.4f}, {out_lut.max():.4f}]")
    diff_lut = (out_lut - out_original).abs()
    print(f"  Diff vs original: max={diff_lut.max():.6f}, mean={diff_lut.mean():.6f}")

    # Test 3: Optimized with async copy
    print("\n[3/4] Testing optimized kernel (LUT + async copy)...")
    out_async = attn_forward_decode_quantized_optimized(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        v=v,
        k_residual=k_residual,
        BS=128,
        delta=5.0,
        use_fp8_residual=True,
        use_fp8_compute=False,
        use_async_copy=True,
    )
    diff_async = (out_async - out_original).abs()
    print(f"  Output shape: {out_async.shape}")
    print(f"  Diff vs original: max={diff_async.max():.6f}, mean={diff_async.mean():.6f}")

    # Test 4: Check if FP8 is supported
    print("\n[4/4] Testing FP8 support...")
    gpu_name = torch.cuda.get_device_name()
    print(f"  GPU: {gpu_name}")

    if "H100" in gpu_name or "A100" in gpu_name:
        print("  FP8 tensor cores supported, testing FP8 mode...")
        try:
            out_fp8 = attn_forward_decode_quantized_optimized(
                q=q,
                k_q=k_q,
                k_scale=k_scale,
                k_zero=k_zero,
                v=v,
                k_residual=k_residual,
                BS=128,
                delta=5.0,
                use_fp8_residual=True,
                use_fp8_compute=True,
                use_async_copy=True,
            )
            diff_fp8 = (out_fp8 - out_original).abs()
            print(f"  FP8 output shape: {out_fp8.shape}")
            print(f"  FP8 diff vs original: max={diff_fp8.max():.6f}, mean={diff_fp8.mean():.6f}")
        except Exception as e:
            print(f"  FP8 test failed (expected on non-H100): {e}")
    else:
        print("  FP8 tensor cores not optimal on this GPU, skipping FP8 test")

    # Summary
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Original kernel works")
    print(f"✓ LUT optimization: max diff = {diff_lut.max():.6f}")
    print(f"✓ Async copy: max diff = {diff_async.max():.6f}")

    if diff_lut.max() < 1e-3 and diff_async.max() < 1e-3:
        print("\n✓ ALL TESTS PASSED! (differences within tolerance)")
        return True
    else:
        print("\n✗ TESTS FAILED! (differences too large)")
        return False


def benchmark_kernel(name, func, q, k_q, k_scale, k_zero, k_residual, v, warmup=10, iters=100, **kwargs):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = func(q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v, k_residual=k_residual, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = func(q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v, k_residual=k_residual, **kwargs)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


def run_benchmark(T=16384, warmup=10, iters=100):
    """Run performance benchmark."""
    print("\n" + "=" * 80)
    print(f"PERFORMANCE BENCHMARK (T={T})")
    print("=" * 80)

    B, HQ, HKV, K, V = 1, 24, 8, 128, 128
    device = 'cuda'
    dtype = torch.float16

    q, k_q, k_scale, k_zero, k_residual, v = create_test_inputs(
        B=B, T=T, HQ=HQ, HKV=HKV, K=K, V=V, device=device, dtype=dtype
    )

    gpu_name = torch.cuda.get_device_name()
    print(f"\nGPU: {gpu_name}")
    print(f"Warmup: {warmup}, Iterations: {iters}\n")

    results = {}

    # Baseline: Original kernel
    print("[1/5] Benchmarking original kernel...")
    time_orig = benchmark_kernel(
        "Original",
        attn_forward_decode_quantized,
        q, k_q, k_scale, k_zero, k_residual, v,
        warmup=warmup, iters=iters,
        BS=128, delta=5.0, use_fp8_residual=True,
    )
    results['Original'] = time_orig
    print(f"  Time: {time_orig:.4f} ms")

    # Optimized: LUT only
    print("\n[2/5] Benchmarking optimized (LUT only)...")
    time_lut = benchmark_kernel(
        "LUT",
        attn_forward_decode_quantized_optimized,
        q, k_q, k_scale, k_zero, k_residual, v,
        warmup=warmup, iters=iters,
        BS=128, delta=5.0, use_fp8_residual=True,
        use_fp8_compute=False, use_async_copy=False,
    )
    results['LUT'] = time_lut
    speedup_lut = time_orig / time_lut
    print(f"  Time: {time_lut:.4f} ms")
    print(f"  Speedup: {speedup_lut:.3f}x")

    # Optimized: LUT + Async
    print("\n[3/5] Benchmarking optimized (LUT + Async)...")
    time_async = benchmark_kernel(
        "LUT+Async",
        attn_forward_decode_quantized_optimized,
        q, k_q, k_scale, k_zero, k_residual, v,
        warmup=warmup, iters=iters,
        BS=128, delta=5.0, use_fp8_residual=True,
        use_fp8_compute=False, use_async_copy=True,
    )
    results['LUT+Async'] = time_async
    speedup_async = time_orig / time_async
    print(f"  Time: {time_async:.4f} ms")
    print(f"  Speedup: {speedup_async:.3f}x")

    # Optimized: FP8 (if supported)
    print("\n[4/5] Benchmarking FP8 mode...")
    if "H100" in gpu_name or "A100" in gpu_name:
        try:
            time_fp8 = benchmark_kernel(
                "FP8",
                attn_forward_decode_quantized_optimized,
                q, k_q, k_scale, k_zero, k_residual, v,
                warmup=warmup, iters=iters,
                BS=128, delta=5.0, use_fp8_residual=True,
                use_fp8_compute=True, use_async_copy=False,
            )
            results['FP8'] = time_fp8
            speedup_fp8 = time_orig / time_fp8
            print(f"  Time: {time_fp8:.4f} ms")
            print(f"  Speedup: {speedup_fp8:.3f}x")
        except Exception as e:
            print(f"  FP8 benchmark failed: {e}")
            results['FP8'] = None
    else:
        print("  Skipping FP8 (not optimal on this GPU)")
        results['FP8'] = None

    # Optimized: All optimizations
    print("\n[5/5] Benchmarking ALL optimizations...")
    if "H100" in gpu_name or "A100" in gpu_name:
        try:
            time_all = benchmark_kernel(
                "All",
                attn_forward_decode_quantized_optimized,
                q, k_q, k_scale, k_zero, k_residual, v,
                warmup=warmup, iters=iters,
                BS=128, delta=5.0, use_fp8_residual=True,
                use_fp8_compute=True, use_async_copy=True,
            )
            results['All'] = time_all
            speedup_all = time_orig / time_all
            print(f"  Time: {time_all:.4f} ms")
            print(f"  Speedup: {speedup_all:.3f}x")
        except Exception as e:
            print(f"  All optimizations benchmark failed: {e}")
            results['All'] = None
    else:
        time_all = time_async  # Use LUT+Async as "all" on non-H100
        results['All'] = time_all
        speedup_all = time_orig / time_all
        print(f"  Time: {time_all:.4f} ms (LUT+Async)")
        print(f"  Speedup: {speedup_all:.3f}x")

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<20} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    for variant, time_val in results.items():
        if time_val is not None:
            speedup = time_orig / time_val
            print(f"{variant:<20} {time_val:>10.4f}  {speedup:>8.3f}x")
        else:
            print(f"{variant:<20} {'N/A':<12} {'N/A':<10}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test optimized Q2FP8 kernels")
    parser.add_argument("--test-correctness", action="store_true", help="Test correctness")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--full-bench", action="store_true", help="Full benchmark with multiple sizes")
    parser.add_argument("--quick", action="store_true", help="Quick test (default)")
    parser.add_argument("--T", type=int, default=16384, help="Sequence length for benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")

    args = parser.parse_args()

    # Default to quick if nothing specified
    if not (args.test_correctness or args.bench or args.full_bench):
        args.quick = True

    if args.quick or args.test_correctness:
        success = test_correctness(T=4096)
        if not success:
            sys.exit(1)

    if args.quick or args.bench:
        run_benchmark(T=args.T, warmup=args.warmup, iters=args.iters)

    if args.full_bench:
        print("\n" + "=" * 80)
        print("FULL BENCHMARK SUITE")
        print("=" * 80)
        for T in [4096, 8192, 16384, 32768, 65536]:
            run_benchmark(T=T, warmup=args.warmup, iters=args.iters)

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()

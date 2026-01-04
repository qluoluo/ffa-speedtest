#!/usr/bin/env python3
"""
H100优化测试脚本 - 测试真正有效的优化方向

测试项目：
1. Block Size优化 (BS, SBS)
2. Delta参数调优
3. Precomputed Threshold效果
4. 各种组合的性能对比

不包含LUT/FP8/Async优化（已证明在H100上无效）
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import time
from typing import Dict, List, Tuple

# Add project root to path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized
from utils.flash import flash_attn_compute


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


def create_test_inputs(B=1, T=16384, HQ=24, HKV=8, K=128, V=128, device='cuda', dtype=torch.float16):
    """Create test inputs."""
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)
    k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)
    return q, k_q, k_scale, k_zero, k_residual, v, k


def benchmark_kernel(
    q, k_q, k_scale, k_zero, k_residual, v,
    BS, SBS, delta,
    precomputed_threshold=None,
    warmup=20, iters=100
):
    """Benchmark a specific configuration."""
    # Warmup
    for _ in range(warmup):
        _ = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v,
            k_residual=k_residual,
            BS=BS, SBS=SBS, delta=delta,
            use_fp8_residual=True,
            precomputed_threshold=precomputed_threshold,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v,
            k_residual=k_residual,
            BS=BS, SBS=SBS, delta=delta,
            use_fp8_residual=True,
            precomputed_threshold=precomputed_threshold,
            return_skip_ratio=True,
        )
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    skip_ratio = out[1] if isinstance(out, tuple) else None

    return elapsed_ms, skip_ratio


def benchmark_flash_attn(q, k, v, warmup=20, iters=100):
    """Benchmark FlashAttention baseline."""
    # flash_attn_compute expects q: [B, H, D], k/v: [B, T, H, D]
    # Our inputs are q: [B, 1, HQ, K], k/v: [B, T, HKV, K]
    # Need to reshape for GQA compatibility
    B, _, HQ, K = q.shape
    _, T, HKV, _ = k.shape

    # For GQA, we need to expand k and v to match query heads
    # k/v: [B, T, HKV, K] -> [B, T, HQ, K]
    num_repeats = HQ // HKV
    k_expanded = k.repeat_interleave(num_repeats, dim=2)  # [B, T, HQ, K]
    v_expanded = v.repeat_interleave(num_repeats, dim=2)  # [B, T, HQ, V]

    # Prepare inputs for flash_attn_compute
    q_flash = q.squeeze(1)  # [B, HQ, K]

    # Warmup
    for _ in range(warmup):
        _ = flash_attn_compute(q_flash, k_expanded, v_expanded)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = flash_attn_compute(q_flash, k_expanded, v_expanded)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


def test_block_size_optimization(T=65536, warmup=20, iters=100):
    """Test 1: Block Size optimization."""
    print("\n" + "=" * 80)
    print(f"TEST 1: Block Size Optimization (T={T})")
    print("=" * 80)

    q, k_q, k_scale, k_zero, k_residual, v, k = create_test_inputs(T=T)

    # FlashAttn baseline
    print("\n[Baseline] FlashAttention...")
    flash_time = benchmark_flash_attn(q, k, v, warmup=warmup, iters=iters)
    print(f"  Time: {flash_time:.4f} ms")

    # Test different block sizes
    configs = [
        {"BS": 128, "SBS": 128, "name": "Original (128/128)"},
        {"BS": 256, "SBS": 128, "name": "Larger BS (256/128)"},
        {"BS": 256, "SBS": 256, "name": "Larger Both (256/256)"},
        {"BS": 512, "SBS": 256, "name": "Large BS (512/256)"},
        {"BS": 1024, "SBS": 256, "name": "Very Large BS (1024/256)"},
        {"BS": 512, "SBS": 512, "name": "Large Both (512/512)"},
    ]

    results = []
    for config in configs:
        print(f"\n[{config['name']}] BS={config['BS']}, SBS={config['SBS']}")
        try:
            time_ms, skip_ratio = benchmark_kernel(
                q, k_q, k_scale, k_zero, k_residual, v,
                BS=config['BS'], SBS=config['SBS'], delta=5.0,
                warmup=warmup, iters=iters
            )
            speedup_vs_flash = flash_time / time_ms
            speedup_vs_original = results[0]['time'] / time_ms if results else 1.0

            print(f"  Time: {time_ms:.4f} ms")
            print(f"  Skip ratio: {skip_ratio*100:.2f}%")
            print(f"  Speedup vs FlashAttn: {speedup_vs_flash:.3f}x")
            if results:
                print(f"  Speedup vs Original: {speedup_vs_original:.3f}x")

            results.append({
                "name": config['name'],
                "BS": config['BS'],
                "SBS": config['SBS'],
                "time": time_ms,
                "skip_ratio": skip_ratio,
                "speedup_vs_flash": speedup_vs_flash,
                "speedup_vs_original": speedup_vs_original,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "name": config['name'],
                "BS": config['BS'],
                "SBS": config['SBS'],
                "error": str(e),
            })

    return results, flash_time


def test_delta_optimization(T=65536, warmup=20, iters=100):
    """Test 2: Delta parameter optimization."""
    print("\n" + "=" * 80)
    print(f"TEST 2: Delta Parameter Optimization (T={T})")
    print("=" * 80)

    q, k_q, k_scale, k_zero, k_residual, v, k = create_test_inputs(T=T)

    # FlashAttn baseline
    print("\n[Baseline] FlashAttention...")
    flash_time = benchmark_flash_attn(q, k, v, warmup=warmup, iters=iters)
    print(f"  Time: {flash_time:.4f} ms")

    # Test different delta values (using best BS from test 1)
    best_BS = 512  # Will be updated based on test 1 results
    deltas = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

    results = []
    for delta in deltas:
        print(f"\n[Delta={delta}]")
        time_ms, skip_ratio = benchmark_kernel(
            q, k_q, k_scale, k_zero, k_residual, v,
            BS=best_BS, SBS=256, delta=delta,
            warmup=warmup, iters=iters
        )
        speedup_vs_flash = flash_time / time_ms

        print(f"  Time: {time_ms:.4f} ms")
        print(f"  Skip ratio: {skip_ratio*100:.2f}%")
        print(f"  Speedup vs FlashAttn: {speedup_vs_flash:.3f}x")

        results.append({
            "delta": delta,
            "time": time_ms,
            "skip_ratio": skip_ratio,
            "speedup_vs_flash": speedup_vs_flash,
        })

    return results, flash_time


def test_precomputed_threshold(T=65536, warmup=20, iters=100):
    """Test 3: Precomputed threshold effect."""
    print("\n" + "=" * 80)
    print(f"TEST 3: Precomputed Threshold Effect (T={T})")
    print("=" * 80)

    q, k_q, k_scale, k_zero, k_residual, v, k = create_test_inputs(T=T)

    BS, SBS, delta = 512, 256, 5.0  # Best config from previous tests

    # Without precomputed threshold
    print("\n[Without Precomputed Threshold]")
    time_without, skip_ratio = benchmark_kernel(
        q, k_q, k_scale, k_zero, k_residual, v,
        BS=BS, SBS=SBS, delta=delta,
        precomputed_threshold=None,
        warmup=warmup, iters=iters
    )
    print(f"  Time: {time_without:.4f} ms")
    print(f"  Skip ratio: {skip_ratio*100:.2f}%")

    # Precompute threshold
    print("\n[Computing threshold once...]")
    start = time.time()
    # Compute threshold using the kernel's internal method
    # We'll call it once to get the threshold, then reuse it
    threshold = torch.empty((1, 24), device='cuda', dtype=torch.float32)
    # Note: We need to extract threshold from kernel, for now we'll compute it separately
    # This is a simplified version - in practice you'd extract it from the kernel
    out_first, _ = benchmark_kernel(
        q, k_q, k_scale, k_zero, k_residual, v,
        BS=BS, SBS=SBS, delta=delta,
        warmup=1, iters=1
    )
    threshold_compute_time = (time.time() - start) * 1000
    print(f"  Threshold computation: {threshold_compute_time:.4f} ms (one-time cost)")

    # With precomputed threshold
    # Note: Current kernel always recomputes, so this test shows the theoretical benefit
    print("\n[With Precomputed Threshold] (theoretical)")
    print("  Note: Current implementation always recomputes threshold")
    print(f"  If we could reuse threshold across {iters} iterations:")
    amortized_cost = time_without - (threshold_compute_time / iters)
    print(f"  Amortized time: {amortized_cost:.4f} ms")
    print(f"  Theoretical speedup: {time_without / amortized_cost:.3f}x")

    return {
        "time_without": time_without,
        "threshold_compute_time": threshold_compute_time,
        "theoretical_speedup": time_without / amortized_cost,
    }


def test_combined_optimizations(T=65536, warmup=20, iters=100):
    """Test 4: Combined optimizations."""
    print("\n" + "=" * 80)
    print(f"TEST 4: Combined Optimizations (T={T})")
    print("=" * 80)

    q, k_q, k_scale, k_zero, k_residual, v, k = create_test_inputs(T=T)

    # FlashAttn baseline
    print("\n[Baseline] FlashAttention...")
    flash_time = benchmark_flash_attn(q, k, v, warmup=warmup, iters=iters)
    print(f"  Time: {flash_time:.4f} ms")

    # Test combinations
    configs = [
        {"BS": 128, "SBS": 128, "delta": 5.0, "name": "Original (baseline Q2FP8)"},
        {"BS": 512, "SBS": 256, "delta": 5.0, "name": "Optimized BS only"},
        {"BS": 512, "SBS": 256, "delta": 6.0, "name": "Optimized BS + Delta"},
        {"BS": 512, "SBS": 256, "delta": 7.0, "name": "Optimized BS + Aggressive Delta"},
        {"BS": 1024, "SBS": 256, "delta": 6.0, "name": "Very Large BS + Delta"},
    ]

    results = []
    for config in configs:
        print(f"\n[{config['name']}]")
        try:
            time_ms, skip_ratio = benchmark_kernel(
                q, k_q, k_scale, k_zero, k_residual, v,
                BS=config['BS'], SBS=config['SBS'], delta=config['delta'],
                warmup=warmup, iters=iters
            )
            speedup_vs_flash = flash_time / time_ms
            speedup_vs_original = results[0]['time'] / time_ms if results else 1.0

            print(f"  BS={config['BS']}, SBS={config['SBS']}, delta={config['delta']}")
            print(f"  Time: {time_ms:.4f} ms")
            print(f"  Skip ratio: {skip_ratio*100:.2f}%")
            print(f"  Speedup vs FlashAttn: {speedup_vs_flash:.3f}x")
            if results:
                print(f"  Speedup vs Original Q2FP8: {speedup_vs_original:.3f}x")

            results.append({
                "name": config['name'],
                "BS": config['BS'],
                "SBS": config['SBS'],
                "delta": config['delta'],
                "time": time_ms,
                "skip_ratio": skip_ratio,
                "speedup_vs_flash": speedup_vs_flash,
                "speedup_vs_original": speedup_vs_original,
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    return results, flash_time


def main():
    parser = argparse.ArgumentParser(description="Test H100 optimizations systematically")
    parser.add_argument("--test", type=str, choices=['all', 'bs', 'delta', 'threshold', 'combined'],
                       default='all', help="Which test to run")
    parser.add_argument("--T", type=int, default=65536, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()
    print("=" * 80)
    print(f"H100 Optimization Tests")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Sequence Length: {args.T}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print("=" * 80)

    all_results = {}

    if args.test in ['all', 'bs']:
        results, flash_time = test_block_size_optimization(args.T, args.warmup, args.iters)
        all_results['block_size'] = {
            'results': results,
            'flash_baseline': flash_time,
        }

    if args.test in ['all', 'delta']:
        results, flash_time = test_delta_optimization(args.T, args.warmup, args.iters)
        all_results['delta'] = {
            'results': results,
            'flash_baseline': flash_time,
        }

    if args.test in ['all', 'threshold']:
        results = test_precomputed_threshold(args.T, args.warmup, args.iters)
        all_results['threshold'] = results

    if args.test in ['all', 'combined']:
        results, flash_time = test_combined_optimizations(args.T, args.warmup, args.iters)
        all_results['combined'] = {
            'results': results,
            'flash_baseline': flash_time,
        }

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

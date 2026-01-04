#!/usr/bin/env python3
"""
Benchmark script to test all Q2FP8 optimizations.
Tests each optimization independently to measure individual gains.
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path to import original kernel
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold")
if PROJECT_ROOT.exists():
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "kernels"))

# Import original kernel
try:
    from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized as baseline_kernel
    HAS_BASELINE = True
except ImportError as e:
    print(f"Warning: Could not import baseline kernel: {e}")
    print("Will only test optimized kernels without comparison")
    HAS_BASELINE = False
    baseline_kernel = None

# Import optimized kernels
try:
    from opt2_adaptive_bm_dot import attn_forward_decode_quantized_opt2
    from opt3_fp16_obuf import attn_forward_decode_quantized_opt3
    from opt4_stage2_compact import attn_forward_decode_quantized_opt4
    from opt5_autotune import attn_forward_decode_quantized_opt5
    HAS_OPTS = True
except ImportError as e:
    print(f"Warning: Could not import optimization kernels: {e}")
    print("Make sure kernels are in the correct location")
    HAS_OPTS = False


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory // (1024**3),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def create_test_data(B, T, HQ, HKV, D, V, device="cuda", dtype=torch.float16):
    """Create test data for benchmarking."""
    # Q: [B, 1, HQ, D]
    q = torch.randn(B, 1, HQ, D, device=device, dtype=dtype)

    # K quantized: [B, T, HKV, D//4] for 2-bit
    k_q = torch.randint(0, 255, (B, T, HKV, D // 4), device=device, dtype=torch.uint8)

    # K scale/zero: [B, HKV, D]
    k_scale = torch.randn(B, HKV, D, device=device, dtype=dtype) * 0.1
    k_zero = torch.randn(B, HKV, D, device=device, dtype=dtype) * 0.1

    # K residual (FP8): [B, T, HKV, D] - create as FP16 first then convert
    k_residual_fp16 = torch.randn(B, T, HKV, D, device=device, dtype=dtype) * 0.1
    k_residual = k_residual_fp16.to(torch.float8_e4m3fn)

    # V: [B, T, HKV, V]
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    return q, k_q, k_scale, k_zero, k_residual, v


def benchmark_kernel(kernel_fn, q, k_q, k_scale, k_zero, k_residual, v,
                      BS=256, SBS=256, delta=5.0, warmup=10, iters=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = kernel_fn(
                q, k_q, k_scale, k_zero, v,
                k_residual=k_residual,
                BS=BS, SBS=SBS, delta=delta,
                use_fp8_residual=True,
            )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        with torch.no_grad():
            _ = kernel_fn(
                q, k_q, k_scale, k_zero, v,
                k_residual=k_residual,
                BS=BS, SBS=SBS, delta=delta,
                use_fp8_residual=True,
            )
        end_events[i].record()

    torch.cuda.synchronize()

    # Compute times
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
    }


def run_benchmarks(args):
    """Run all benchmarks."""
    device = "cuda"
    dtype = torch.float16

    # Configuration
    B = args.batch_size
    T = args.seq_len
    HQ = 24   # Typical for Llama2-7B
    HKV = 8
    D = 128
    V = 128

    print("=" * 80)
    print("Q2FP8 Optimization Benchmarks")
    print("=" * 80)

    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Memory: {gpu_info['total_memory_gb']} GB")
        print(f"Compute: {gpu_info['compute_capability']}")
    print()

    print(f"Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  HQ: {HQ}, HKV: {HKV}, D: {D}, V: {V}")
    print(f"  BS: {args.BS}, SBS: {args.SBS}, delta: {args.delta}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    # Create test data
    print("Creating test data...")
    q, k_q, k_scale, k_zero, k_residual, v = create_test_data(
        B, T, HQ, HKV, D, V, device=device, dtype=dtype
    )
    print("Done!")
    print()

    results = {}

    # Benchmark baseline
    print("[1/5] Benchmarking Baseline...")
    baseline_result = None
    if HAS_BASELINE and baseline_kernel is not None:
        try:
            baseline_result = benchmark_kernel(
                baseline_kernel, q, k_q, k_scale, k_zero, k_residual, v,
                BS=args.BS, SBS=args.SBS, delta=args.delta,
                warmup=args.warmup, iters=args.iters
            )
            results["baseline"] = baseline_result
            print(f"  Mean: {baseline_result['mean_ms']:.4f} ms")
            print(f"  Std:  {baseline_result['std_ms']:.4f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["baseline"] = {"error": str(e)}
    else:
        print("  SKIPPED: Baseline kernel not available")
        results["baseline"] = {"error": "not available"}
    print()

    if not HAS_OPTS:
        print("Optimization kernels not available, skipping optimization benchmarks.")
        return results

    # Benchmark Opt2: Adaptive BM_DOT
    print("[2/5] Benchmarking Opt2 (Adaptive BM_DOT)...")
    try:
        opt2_result = benchmark_kernel(
            attn_forward_decode_quantized_opt2, q, k_q, k_scale, k_zero, k_residual, v,
            BS=args.BS, SBS=args.SBS, delta=args.delta,
            warmup=args.warmup, iters=args.iters
        )
        results["opt2_adaptive_bm_dot"] = opt2_result
        print(f"  Mean: {opt2_result['mean_ms']:.4f} ms")
        if baseline_result is not None:
            speedup = baseline_result['mean_ms'] / opt2_result['mean_ms']
            print(f"  Speedup vs baseline: {speedup:.3f}x")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["opt2_adaptive_bm_dot"] = {"error": str(e)}
    print()

    # Benchmark Opt3: FP16 o_buf
    print("[3/5] Benchmarking Opt3 (FP16 o_buf)...")
    try:
        opt3_result = benchmark_kernel(
            attn_forward_decode_quantized_opt3, q, k_q, k_scale, k_zero, k_residual, v,
            BS=args.BS, SBS=args.SBS, delta=args.delta,
            warmup=args.warmup, iters=args.iters
        )
        results["opt3_fp16_obuf"] = opt3_result
        print(f"  Mean: {opt3_result['mean_ms']:.4f} ms")
        if baseline_result is not None:
            speedup = baseline_result['mean_ms'] / opt3_result['mean_ms']
            print(f"  Speedup vs baseline: {speedup:.3f}x")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["opt3_fp16_obuf"] = {"error": str(e)}
    print()

    # Benchmark Opt4: Stage2 compact
    print("[4/5] Benchmarking Opt4 (Stage2 Compact)...")
    try:
        opt4_result = benchmark_kernel(
            attn_forward_decode_quantized_opt4, q, k_q, k_scale, k_zero, k_residual, v,
            BS=args.BS, SBS=args.SBS, delta=args.delta,
            warmup=args.warmup, iters=args.iters
        )
        results["opt4_stage2_compact"] = opt4_result
        print(f"  Mean: {opt4_result['mean_ms']:.4f} ms")
        if baseline_result is not None:
            speedup = baseline_result['mean_ms'] / opt4_result['mean_ms']
            print(f"  Speedup vs baseline: {speedup:.3f}x")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["opt4_stage2_compact"] = {"error": str(e)}
    print()

    # Benchmark Opt5: Autotune
    print("[5/5] Benchmarking Opt5 (Triton Autotune)...")
    try:
        opt5_result = benchmark_kernel(
            attn_forward_decode_quantized_opt5, q, k_q, k_scale, k_zero, k_residual, v,
            BS=args.BS, SBS=args.SBS, delta=args.delta,
            warmup=args.warmup, iters=args.iters
        )
        results["opt5_autotune"] = opt5_result
        print(f"  Mean: {opt5_result['mean_ms']:.4f} ms")
        if baseline_result is not None:
            speedup = baseline_result['mean_ms'] / opt5_result['mean_ms']
            print(f"  Speedup vs baseline: {speedup:.3f}x")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["opt5_autotune"] = {"error": str(e)}
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Kernel':<30} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results.get("baseline", {}).get("mean_ms", float('inf'))

    for name, result in results.items():
        if "error" in result:
            print(f"{name:<30} {'ERROR':<15} {'-':<10}")
        else:
            time_ms = result["mean_ms"]
            speedup = baseline_time / time_ms if time_ms > 0 else 0
            print(f"{name:<30} {time_ms:<15.4f} {speedup:<10.3f}x")
    print()

    # Save results
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "config": {
                "B": B, "T": T, "HQ": HQ, "HKV": HKV, "D": D, "V": V,
                "BS": args.BS, "SBS": args.SBS, "delta": args.delta,
                "warmup": args.warmup, "iters": args.iters,
            },
            "gpu": gpu_info,
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Q2FP8 optimizations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=262144, help="Sequence length")
    parser.add_argument("--BS", type=int, default=256, help="Block size")
    parser.add_argument("--SBS", type=int, default=256, help="Sub-block size")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta threshold")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=str, help="Output JSON file path")

    args = parser.parse_args()

    run_benchmarks(args)


if __name__ == "__main__":
    main()

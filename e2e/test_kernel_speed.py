#!/usr/bin/env python3
"""
直接测试 Q2FP8 attention kernel 的性能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "q2fp8" / "attn_kernel"))

import torch
import time
from attn_q2fp8_sym_lr64_compact import attn_forward_decode_quantized, CUDAGraphDecodeRunnerQ2FP8


def benchmark_kernel(
    B=1, T=262144, HQ=24, HKV=8, K=128, V=128,
    BS=128, SBS=128, delta=5.0,
    use_cudagraph=False,
    warmup=100, iters=500,
    return_skip_ratio=False
):
    """Benchmark Q2FP8 kernel performance"""
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Prepare inputs
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)

    # Quantized K cache (2-bit packed)
    K_PACKED = (K + 3) // 4  # 2-bit packing
    k_q = torch.randint(0, 255, (B, T, HKV, K_PACKED), device=device, dtype=torch.uint8)

    # K scale
    k_scale = torch.randn(B, HKV, K, device=device, dtype=torch.float32).abs() * 0.01

    # FP8 residual
    k_residual = torch.randn(B, T, HKV, K, device=device, dtype=torch.float16) * 0.001

    # V cache
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    print(f"Testing Q2FP8 kernel:")
    print(f"  Shape: B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}")
    print(f"  BS={BS}, SBS={SBS}, delta={delta}")
    print(f"  Use CUDAGraph: {use_cudagraph}")

    if use_cudagraph:
        # Test with CUDAGraph
        print(f"\n[CUDAGraph] Creating runner...")
        runner = CUDAGraphDecodeRunnerQ2FP8(
            q, k_q, k_scale, v,
            k_residual=k_residual,
            k_bits=2,
            BS=BS,
            SBS=SBS,
            delta=delta,
            use_fp8_residual=True,
            warmup=warmup,
        )

        # Get skip ratio
        if return_skip_ratio:
            _, skip_ratio = attn_forward_decode_quantized(
                q, k_q, k_scale, v,
                k_residual=k_residual,
                k_bits=2,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
                return_skip_ratio=True,
            )
            print(f"  Skip ratio: {skip_ratio:.4f}")

        print(f"[CUDAGraph] Warming up {warmup} iterations...")
        for _ in range(warmup):
            _ = runner.replay_only()
        torch.cuda.synchronize()

        print(f"[CUDAGraph] Running {iters} iterations...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = runner.replay_only()
        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        print(f"\n[CUDAGraph] Average time: {elapsed_ms:.4f} ms")

    else:
        # Test without CUDAGraph
        # Get skip ratio
        if return_skip_ratio:
            _, skip_ratio = attn_forward_decode_quantized(
                q, k_q, k_scale, v,
                k_residual=k_residual,
                k_bits=2,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
                return_skip_ratio=True,
            )
            print(f"  Skip ratio: {skip_ratio:.4f}")

        print(f"\n[Standard] Warming up {warmup} iterations...")
        for _ in range(warmup):
            _ = attn_forward_decode_quantized(
                q, k_q, k_scale, v,
                k_residual=k_residual,
                k_bits=2,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
            )
        torch.cuda.synchronize()

        print(f"[Standard] Running {iters} iterations...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = attn_forward_decode_quantized(
                q, k_q, k_scale, v,
                k_residual=k_residual,
                k_bits=2,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
            )
        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        print(f"\n[Standard] Average time: {elapsed_ms:.4f} ms")

    return elapsed_ms


if __name__ == "__main__":
    print("=" * 70)
    print("Q2FP8 Kernel Performance Test")
    print("=" * 70)

    # Test configurations matching your previous benchmark
    T = 262144  # 256K sequence length

    print("\n" + "=" * 70)
    print("Test 1: Standard kernel (no CUDAGraph)")
    print("=" * 70)
    time_standard = benchmark_kernel(T=T, use_cudagraph=False, warmup=100, iters=500, return_skip_ratio=True)

    print("\n" + "=" * 70)
    print("Test 2: CUDAGraph kernel")
    print("=" * 70)
    time_cudagraph = benchmark_kernel(T=T, use_cudagraph=True, warmup=100, iters=500, return_skip_ratio=True)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Standard kernel:   {time_standard:.4f} ms")
    print(f"CUDAGraph kernel:  {time_cudagraph:.4f} ms")
    print(f"Speedup:           {time_standard/time_cudagraph:.2f}x")
    print("\nNote: Random data has low skip ratio (~0.0), real data has high skip ratio (~0.998)")
    print("This explains the 4-6x performance difference with ffa-q2fp8-threshold-opt results.")

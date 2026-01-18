#!/usr/bin/env python3
"""
Benchmark script for Paged Q2FP8 Attention Kernel.

This script benchmarks the paged quantized attention kernel against
Flash Attention for various sequence lengths.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import triton

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from attn_kernel import attn_forward_decode_quantized_paged
from utils.bench import benchmark
from utils.cache import convert_to_paged_format, to_k_str


def create_test_data(
    B: int,
    T: int,
    HQ: int,
    HKV: int,
    K: int,
    V: int,
    page_size: int,
    k_bits: int = 2,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Create test data for paged attention benchmark."""
    # Query: [B, 1, HQ, K]
    q = torch.randn(B, 1, HQ, K, dtype=dtype, device=device)

    # Create contiguous KV cache first
    vals_per_byte = 8 // k_bits
    K_packed = (K + vals_per_byte - 1) // vals_per_byte

    # Quantized K: [B, T, HKV, K_packed]
    k_q_cont = torch.randint(0, 256, (B, T, HKV, K_packed), dtype=torch.uint8, device=device)

    # V: [B, T, HKV, V]
    v_cont = torch.randn(B, T, HKV, V, dtype=dtype, device=device)

    # Convert to paged format
    k_q_paged, v_paged, page_table, seq_lens = convert_to_paged_format(
        k_q_cont, v_cont, page_size, device
    )

    # Scale: [B, HKV, K]
    k_scale = torch.randn(B, HKV, K, dtype=torch.float32, device=device) * 0.1

    return q, k_q_paged, k_scale, v_paged, page_table, seq_lens


def run_benchmark(
    B: int = 1,
    T_max: int = 131072,
    HQ: int = 32,
    HKV: int = 8,
    K: int = 128,
    V: int = 128,
    page_size: int = 16,
    BS: int = 128,
    SBS: int = 128,
    delta: float = 5.0,
    step: int = 4096,
    iters: int = 50,
    warmup: int = 10,
    device: str = "cuda",
):
    """Run benchmark for various sequence lengths."""
    print(f"Benchmarking Paged Q2FP8 Attention Kernel")
    print(f"  B={B}, HQ={HQ}, HKV={HKV}, K={K}, V={V}")
    print(f"  page_size={page_size}, BS={BS}, SBS={SBS}, delta={delta}")
    print(f"  T_max={to_k_str(T_max)}, step={step}")
    print(f"  iters={iters}, warmup={warmup}")
    print("-" * 60)

    lengths = list(range(step, T_max + 1, step))
    paged_ms_list = []
    skip_ratios = []

    for T in lengths:
        print(f"T={T} ({to_k_str(T)})", end=" ... ", flush=True)

        # Create test data
        q, k_q_paged, k_scale, v_paged, page_table, seq_lens = create_test_data(
            B, T, HQ, HKV, K, V, page_size, device=device
        )

        # Benchmark paged kernel
        def run_paged():
            return attn_forward_decode_quantized_paged(
                q=q,
                k_q=k_q_paged,
                k_scale=k_scale,
                v=v_paged,
                page_table=page_table,
                seq_lens=seq_lens,
                k_bits=2,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=False,
            )

        paged_ms = benchmark(run_paged, iters=iters, warmup=warmup)
        paged_ms_list.append(paged_ms)

        # Get skip ratio
        _, skip_ratio = attn_forward_decode_quantized_paged(
            q=q,
            k_q=k_q_paged,
            k_scale=k_scale,
            v=v_paged,
            page_table=page_table,
            seq_lens=seq_lens,
            k_bits=2,
            BS=BS,
            SBS=SBS,
            delta=delta,
            use_fp8_residual=False,
            return_skip_ratio=True,
        )
        skip_ratios.append(skip_ratio)

        print(f"paged={paged_ms:.3f}ms, skip_ratio={skip_ratio*100:.1f}%")

        # Cleanup
        del q, k_q_paged, k_scale, v_paged, page_table, seq_lens
        torch.cuda.empty_cache()

    print("-" * 60)
    print("Benchmark complete!")

    return lengths, paged_ms_list, skip_ratios


def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged Q2FP8 Attention Kernel")
    parser.add_argument("--batch-size", "-B", type=int, default=1, help="Batch size")
    parser.add_argument("--t-max", "-T", type=int, default=131072, help="Max sequence length")
    parser.add_argument("--hq", type=int, default=32, help="Number of query heads")
    parser.add_argument("--hkv", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--k-dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--v-dim", type=int, default=128, help="Value dimension")
    parser.add_argument("--page-size", type=int, default=16, help="Page size for paged KV cache")
    parser.add_argument("--bs", type=int, default=128, help="Block size")
    parser.add_argument("--sbs", type=int, default=128, help="Sub-block size")
    parser.add_argument("--delta", type=float, default=5.0, help="Threshold delta")
    parser.add_argument("--step", type=int, default=4096, help="Step size for sequence lengths")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    args = parser.parse_args()

    run_benchmark(
        B=args.batch_size,
        T_max=args.t_max,
        HQ=args.hq,
        HKV=args.hkv,
        K=args.k_dim,
        V=args.v_dim,
        page_size=args.page_size,
        BS=args.bs,
        SBS=args.sbs,
        delta=args.delta,
        step=args.step,
        iters=args.iters,
        warmup=args.warmup,
        device=args.device,
    )


if __name__ == "__main__":
    main()

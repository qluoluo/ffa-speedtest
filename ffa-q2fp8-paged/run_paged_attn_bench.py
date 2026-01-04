"""
Benchmark script for paged Q2FP8 attention.

Compares performance and memory usage of paged vs. non-paged implementations.
"""

import argparse
import time
from typing import Dict

import torch

from attn_kernel.paged_attn import paged_attn_forward_decode
from attn_kernel.page_quant import quantize_k_page_q2fp8, quantize_k_multi_pages
from e2e.paged_q2fp8_cache import PagedQ2FP8Cache


def benchmark_paged_attention(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    page_size: int = 128,
    delta: float = 5.0,
    use_threshold: bool = True,
    warmup: int = 10,
    iters: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark paged attention performance.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads_q: Number of query heads
        num_heads_kv: Number of KV heads
        head_dim: Head dimension
        page_size: Page size
        delta: Threshold delta for pruning
        use_threshold: Whether to use threshold pruning
        warmup: Warmup iterations
        iters: Benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    device = torch.device(device)
    dtype = torch.float16

    # Create cache
    max_pages = (seq_len + page_size - 1) // page_size * batch_size + 10
    cache = PagedQ2FP8Cache(
        page_size=page_size,
        max_pages=max_pages,
        max_batch_size=batch_size,
        num_layers=1,
        use_fp8_residual=True,
        device=device,
        dtype=dtype,
    )

    # Fill cache with random data
    print(f"Filling cache with {seq_len} tokens...")
    key_states = torch.randn(1, seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
    value_states = torch.randn(1, seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)

    for b in range(batch_size):
        cache.update(key_states, value_states, layer_idx=0, batch_idx=b)

    layer0 = cache.get_layer(0)

    # Create query
    q = torch.randn(batch_size, 1, num_heads_q, head_dim, device=device, dtype=dtype)

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Num pages per batch: {layer0.num_pages_per_batch[:batch_size]}")
    print(f"Total physical pages: {cache.next_free_page}")

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        output = paged_attn_forward_decode(
            q=q,
            page_table_k=layer0.page_table_k[:batch_size],
            k_pages_q=layer0.k_pages_q,
            k_pages_scale=layer0.k_pages_scale,
            k_pages_zero=layer0.k_pages_zero,
            k_pages_residual=layer0.k_pages_residual,
            v_pages=layer0.v_pages,
            seq_lens=layer0.seq_lens[:batch_size],
            page_size=page_size,
            delta=delta,
            use_threshold_pruning=use_threshold,
        )
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iters} iterations)...")
    start = time.perf_counter()
    for _ in range(iters):
        output, stats = paged_attn_forward_decode(
            q=q,
            page_table_k=layer0.page_table_k[:batch_size],
            k_pages_q=layer0.k_pages_q,
            k_pages_scale=layer0.k_pages_scale,
            k_pages_zero=layer0.k_pages_zero,
            k_pages_residual=layer0.k_pages_residual,
            v_pages=layer0.v_pages,
            seq_lens=layer0.seq_lens[:batch_size],
            page_size=page_size,
            delta=delta,
            use_threshold_pruning=use_threshold,
            return_stats=True,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iters * 1000

    # Memory usage
    if device.type == "cuda":
        # K cache memory
        k_q_mem = layer0.k_pages_q.element_size() * layer0.k_pages_q.numel()
        k_scale_mem = layer0.k_pages_scale.element_size() * layer0.k_pages_scale.numel()
        k_zero_mem = layer0.k_pages_zero.element_size() * layer0.k_pages_zero.numel()
        k_res_mem = (
            layer0.k_pages_residual.element_size() * layer0.k_pages_residual.numel()
            if layer0.k_pages_residual is not None
            else 0
        )
        v_mem = layer0.v_pages.element_size() * layer0.v_pages.numel()

        total_mem_mb = (k_q_mem + k_scale_mem + k_zero_mem + k_res_mem + v_mem) / (1024 ** 2)

        # Baseline memory (uncompressed)
        baseline_k_mem = batch_size * seq_len * num_heads_kv * head_dim * 2  # fp16
        baseline_v_mem = batch_size * seq_len * num_heads_kv * head_dim * 2  # fp16
        baseline_mem_mb = (baseline_k_mem + baseline_v_mem) / (1024 ** 2)

        compression_ratio = baseline_mem_mb / total_mem_mb
    else:
        total_mem_mb = 0.0
        baseline_mem_mb = 0.0
        compression_ratio = 1.0

    results = {
        "avg_time_ms": avg_time_ms,
        "total_mem_mb": total_mem_mb,
        "baseline_mem_mb": baseline_mem_mb,
        "compression_ratio": compression_ratio,
        "prune_ratio": stats.get("prune_ratio", 0.0),
        "kept_pages": stats.get("kept_pages", 0),
        "total_pages": stats.get("total_pages", 0),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark paged Q2FP8 attention")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num-heads-q", type=int, default=32, help="Number of query heads")
    parser.add_argument("--num-heads-kv", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--page-size", type=int, default=128, help="Page size")
    parser.add_argument("--delta", type=float, default=5.0, help="Threshold delta")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-threshold", action="store_true", help="Disable threshold pruning")
    args = parser.parse_args()

    print("=" * 80)
    print("Paged Q2FP8 Attention Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num heads Q/KV: {args.num_heads_q}/{args.num_heads_kv}")
    print(f"  Head dimension: {args.head_dim}")
    print(f"  Page size: {args.page_size}")
    print(f"  Delta: {args.delta}")
    print(f"  Threshold pruning: {not args.no_threshold}")
    print(f"  Device: {args.device}")
    print("=" * 80)

    results = benchmark_paged_attention(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        head_dim=args.head_dim,
        page_size=args.page_size,
        delta=args.delta,
        use_threshold=not args.no_threshold,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Average time: {results['avg_time_ms']:.3f} ms")
    print(f"Total memory: {results['total_mem_mb']:.2f} MB")
    print(f"Baseline memory (FP16): {results['baseline_mem_mb']:.2f} MB")
    print(f"Compression ratio: {results['compression_ratio']:.2f}x")
    if not args.no_threshold:
        print(f"Pages kept: {results['kept_pages']} / {results['total_pages']}")
        print(f"Prune ratio: {results['prune_ratio']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Benchmark script for paged Q2FP8 attention - 256K long context test.
Similar to ffa-q2fp8-threshold but for paged implementation.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from attn_kernel.paged_attn import paged_attn_forward_decode
from e2e.paged_q2fp8_cache import PagedQ2FP8Cache


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = props.name.strip()
    total_mem_gb = math.ceil(props.total_memory / (1024**3))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "gpu"
    tag = f"{safe_name}_{total_mem_gb}GB"
    return tag, name, total_mem_gb, device_idx


def to_k_str(length: int) -> str:
    """Convert length to K string (e.g., 262144 -> 256K)."""
    if length >= 1024:
        return f"{length // 1024}K"
    return str(length)


def benchmark_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 10,
    iters: int = 100,
) -> Optional[float]:
    """Benchmark Flash Attention as baseline."""
    try:
        from flash_attn import flash_attn_func

        # Warmup
        for _ in range(warmup):
            _ = flash_attn_func(
                q.transpose(1, 2),  # [B, seq, H, D]
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=False,
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            _ = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=False,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / iters * 1000
        return avg_time_ms

    except ImportError:
        print("[Warning] flash-attn not installed, skipping FlashAttention baseline")
        return None


def benchmark_paged_attn(
    cache: PagedQ2FP8Cache,
    layer_idx: int,
    batch_idx: int,
    q: torch.Tensor,
    page_size: int,
    delta: float,
    warmup: int = 10,
    iters: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark paged attention.

    Returns:
        avg_time_ms: Average time in milliseconds
        prune_ratio: Pruning ratio (0-1)
    """
    layer_cache = cache.get_layer(layer_idx)

    # Warmup
    for _ in range(warmup):
        _ = paged_attn_forward_decode(
            q=q,
            page_table_k=layer_cache.page_table_k[batch_idx:batch_idx+1],
            k_pages_q=layer_cache.k_pages_q,
            k_pages_scale=layer_cache.k_pages_scale,
            k_pages_zero=layer_cache.k_pages_zero,
            k_pages_residual=layer_cache.k_pages_residual,
            v_pages=layer_cache.v_pages,
            seq_lens=layer_cache.seq_lens[batch_idx:batch_idx+1],
            page_size=page_size,
            delta=delta,
            use_threshold_pruning=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    stats_list = []
    for _ in range(iters):
        _, stats = paged_attn_forward_decode(
            q=q,
            page_table_k=layer_cache.page_table_k[batch_idx:batch_idx+1],
            k_pages_q=layer_cache.k_pages_q,
            k_pages_scale=layer_cache.k_pages_scale,
            k_pages_zero=layer_cache.k_pages_zero,
            k_pages_residual=layer_cache.k_pages_residual,
            v_pages=layer_cache.v_pages,
            seq_lens=layer_cache.seq_lens[batch_idx:batch_idx+1],
            page_size=page_size,
            delta=delta,
            use_threshold_pruning=True,
            return_stats=True,
        )
        stats_list.append(stats)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iters * 1000
    avg_prune_ratio = sum(s['prune_ratio'] for s in stats_list) / len(stats_list)

    return avg_time_ms, avg_prune_ratio


def plot_speed_curve(
    x_lengths: List[int],
    paged_ms_list: List[float],
    flash_ms_list: Optional[List[float]],
    skip_ratios: List[float],
    max_length: int,
    page_size: int,
    delta: float,
    gpu_label: str,
    out_path: Path,
):
    """Plot performance curve."""
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot latency
    line_paged, = ax1.plot(
        x_lengths, paged_ms_list,
        label="Paged Q2FP8",
        marker="o",
        markersize=3,
        color="tab:blue",
        linewidth=2,
    )

    lines = [line_paged]
    labels = ["Paged Q2FP8"]

    if flash_ms_list is not None:
        line_flash, = ax1.plot(
            x_lengths, flash_ms_list,
            label="FlashAttn (FP16)",
            marker="s",
            markersize=3,
            color="tab:orange",
            linewidth=2,
        )
        lines.append(line_flash)
        labels.append("FlashAttn (FP16)")

    ax1.set_xlabel("Sequence Length (T)", fontsize=12)
    ax1.set_ylabel("Latency per Decode (ms)", fontsize=12)
    ax1.set_title(
        f"Paged Q2FP8 Attention Performance\n"
        f"(Max={to_k_str(max_length)}, PageSize={page_size}, Delta={delta})",
        fontsize=14,
        fontweight='bold',
    )
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Plot skip ratio on secondary axis
    ax2 = ax1.twinx()
    skip_pct = [sr * 100.0 for sr in skip_ratios]
    line_skip, = ax2.plot(
        x_lengths,
        skip_pct,
        label="Prune Ratio (%)",
        color="tab:green",
        linestyle="--",
        marker="x",
        markersize=4,
        linewidth=1.5,
    )
    ax2.set_ylabel("Prune Ratio (%)", fontsize=12, color="tab:green")
    ax2.tick_params(axis='y', labelcolor="tab:green")
    ax2.set_ylim(0, 100)

    lines.append(line_skip)
    labels.append("Prune Ratio (%)")

    # Add GPU info
    ax1.text(
        0.02, 0.98,
        f"GPU: {gpu_label}",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Legend
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] Saved plot to {out_path}")


def run_benchmark(
    max_length: int = 262144,  # 256K
    step: int = 16384,  # 16K
    page_size: int = 128,
    delta: float = 5.0,
    num_heads_q: int = 32,
    num_heads_kv: int = 8,
    head_dim: int = 128,
    warmup: int = 10,
    iters: int = 50,
    skip_flash: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Run complete benchmark."""
    device = torch.device(device)

    # Get GPU info
    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    # Configuration
    print(f"\n{'='*80}")
    print("Benchmark Configuration")
    print(f"{'='*80}")
    print(f"  Max length: {to_k_str(max_length)} ({max_length})")
    print(f"  Step: {to_k_str(step)} ({step})")
    print(f"  Page size: {page_size}")
    print(f"  Delta: {delta}")
    print(f"  Num heads Q/KV: {num_heads_q}/{num_heads_kv}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Warmup/Iters: {warmup}/{iters}")
    print(f"  Dtype: {dtype}")
    print(f"{'='*80}\n")

    # Create cache
    max_pages = (max_length + page_size - 1) // page_size + 10
    cache = PagedQ2FP8Cache(
        page_size=page_size,
        max_pages=max_pages,
        max_batch_size=1,
        num_layers=1,
        use_fp8_residual=True,
        device=device,
        dtype=dtype,
    )

    # Generate test lengths
    lengths = list(range(step, max_length, step)) + [max_length]
    print(f"[Info] Testing {len(lengths)} lengths: {to_k_str(lengths[0])} to {to_k_str(lengths[-1])}")

    # Results storage
    x_lengths = []
    paged_ms_list = []
    flash_ms_list = [] if not skip_flash else None
    skip_ratios = []

    # Benchmark loop
    print(f"\n{'='*80}")
    print("Running Benchmarks")
    print(f"{'='*80}\n")

    for L in tqdm(lengths, desc="Benchmarking"):
        # Generate random data for this length
        key_states = torch.randn(1, L, num_heads_kv, head_dim, device=device, dtype=dtype)
        value_states = torch.randn(1, L, num_heads_kv, head_dim, device=device, dtype=dtype)
        query = torch.randn(1, 1, num_heads_q, head_dim, device=device, dtype=dtype)

        # Reset cache and fill with new data
        cache.reset()
        cache.update(key_states, value_states, layer_idx=0, batch_idx=0)

        # Benchmark paged attention
        paged_ms, prune_ratio = benchmark_paged_attn(
            cache=cache,
            layer_idx=0,
            batch_idx=0,
            q=query,
            page_size=page_size,
            delta=delta,
            warmup=warmup,
            iters=iters,
        )

        # Benchmark Flash Attention
        flash_ms = None
        if not skip_flash:
            # Reshape for flash attention
            q_flash = query.squeeze(1)  # [1, HQ, K]
            k_flash = key_states.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, L, HKV, K] -> [1, HKV, L, K]
            v_flash = value_states.squeeze(0).transpose(0, 1).unsqueeze(0)

            flash_ms = benchmark_flash_attention(
                q=q_flash.unsqueeze(1),  # [1, 1, HQ, K]
                k=k_flash.transpose(1, 2),  # [1, L, HKV, K]
                v=v_flash.transpose(1, 2),
                warmup=warmup,
                iters=iters,
            )

        # Store results
        x_lengths.append(L)
        paged_ms_list.append(paged_ms)
        if flash_ms is not None:
            flash_ms_list.append(flash_ms)
        skip_ratios.append(prune_ratio)

        # Print progress
        if L == lengths[0] or L == max_length or L % (step * 4) == 0:
            speedup = f"{flash_ms / paged_ms:.2f}x" if flash_ms is not None else "N/A"
            print(f"\n  Length {to_k_str(L):>6}: Paged={paged_ms:>7.3f}ms, "
                  f"Flash={flash_ms:>7.3f}ms" if flash_ms is not None else f", Speedup={speedup}, "
                  f"Prune={prune_ratio*100:.1f}%")

    # Save results
    output_dir = Path(__file__).parent / "plot" / "paged_q2fp8_256k" / gpu_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    results = {
        "x_lengths": x_lengths,
        "paged_ms_list": paged_ms_list,
        "flash_ms_list": flash_ms_list,
        "skip_ratios": skip_ratios,
        "config": {
            "max_length": max_length,
            "step": step,
            "page_size": page_size,
            "delta": delta,
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "warmup": warmup,
            "iters": iters,
            "dtype": str(dtype),
        },
        "gpu": {
            "tag": gpu_tag,
            "name": gpu_name,
            "memory_gb": gpu_mem_gb,
        },
    }

    json_path = output_dir / f"results_max{to_k_str(max_length)}_page{page_size}_delta{delta}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Info] Saved results to {json_path}")

    # Plot
    plot_path = output_dir / f"performance_max{to_k_str(max_length)}_page{page_size}_delta{delta}.png"
    plot_speed_curve(
        x_lengths=x_lengths,
        paged_ms_list=paged_ms_list,
        flash_ms_list=flash_ms_list,
        skip_ratios=skip_ratios,
        max_length=max_length,
        page_size=page_size,
        delta=delta,
        gpu_label=gpu_label,
        out_path=plot_path,
    )

    # Print summary
    print(f"\n{'='*80}")
    print("Benchmark Summary")
    print(f"{'='*80}")
    print(f"  Total lengths tested: {len(x_lengths)}")
    print(f"  Paged Q2FP8 avg time: {sum(paged_ms_list)/len(paged_ms_list):.3f} ms")
    if flash_ms_list:
        print(f"  FlashAttn avg time: {sum(flash_ms_list)/len(flash_ms_list):.3f} ms")
        avg_speedup = sum(f/p for f, p in zip(flash_ms_list, paged_ms_list)) / len(paged_ms_list)
        print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Average prune ratio: {sum(skip_ratios)/len(skip_ratios)*100:.2f}%")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark paged Q2FP8 attention for long context")
    parser.add_argument("--max-length", type=int, default=262144, help="Maximum sequence length (default: 256K)")
    parser.add_argument("--step", type=int, default=16384, help="Step size for length sweep (default: 16K)")
    parser.add_argument("--page-size", type=int, default=128, help="Page size (default: 128)")
    parser.add_argument("--delta", type=float, default=5.0, help="Threshold delta for pruning (default: 5.0)")
    parser.add_argument("--num-heads-q", type=int, default=32, help="Number of query heads (default: 32)")
    parser.add_argument("--num-heads-kv", type=int, default=8, help="Number of KV heads (default: 8)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations (default: 50)")
    parser.add_argument("--skip-flash", action="store_true", help="Skip FlashAttention baseline")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Data type")
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    run_benchmark(
        max_length=args.max_length,
        step=args.step,
        page_size=args.page_size,
        delta=args.delta,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        head_dim=args.head_dim,
        warmup=args.warmup,
        iters=args.iters,
        skip_flash=args.skip_flash,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()

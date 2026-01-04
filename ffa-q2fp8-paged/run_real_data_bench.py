"""
Benchmark paged Q2FP8 attention using real recorded data.
Similar to ffa-q2fp8-threshold/run_attn_bench_q2_cudagraph.py but for paged version.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Load function (copied from ffa-q2fp8-threshold/utils/load.py)
def load_qkvh(load_dir: str, device='cpu', start_layer: int = 0, max_length: Optional[int] = None):
    """Load Q/K/V data from recorded layer data."""
    load_root = Path(load_dir)
    dirname_list = sorted([x for x in load_root.iterdir() if x.is_dir() and x.name.startswith("layer")],
                          key=lambda x: int(x.name.split("_")[1]))
    layer_num = len(dirname_list)

    assert [p.name for p in dirname_list] == [
        f"layer_{i}" for i in range(layer_num)
    ], "Layer directories must be named layer_0, layer_1, ..."

    if not (0 <= start_layer < layer_num):
        raise ValueError(f"start_layer must be in [0, {layer_num - 1}], got {start_layer}")

    def _truncate_tensor(t: torch.Tensor):
        if max_length is None or max_length <= 0:
            return t
        if t.dim() >= 3:
            return t[..., :max_length, :]
        if t.dim() == 2:
            return t[:, :max_length]
        return t

    for i in range(start_layer, layer_num):
        layer_dir = load_root / f"layer_{i}"
        load_data_list = ["q_rope", "k_rope", "q_unrope", "k_unrope", "v", "h"]
        data = {}
        for data_name in load_data_list:
            data_path = layer_dir / f"{data_name}.pt"
            tensor = torch.load(data_path, weights_only=True, map_location=device)
            data[data_name] = _truncate_tensor(tensor)
        yield data

# Define to_k_str locally
def to_k_str(length: int) -> str:
    """Convert length to K string."""
    if length >= 1024:
        return f"{length // 1024}K"
    return str(length)

from attn_kernel.paged_attn import paged_attn_forward_decode
from e2e.paged_q2fp8_cache import PagedQ2FP8Cache


# Default data path (same as original)
EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Paged Q2FP8 with real data")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max length to test (default: use full recorded length)",
    )
    p.add_argument("--step", type=int, default=1024, help="Step size for length sweep")
    p.add_argument("--iters", type=int, default=500, help="Benchmark iterations")
    p.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = props.name.strip()
    total_mem_gb = math.ceil(props.total_memory / (1024**3))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "gpu"
    tag = f"{safe_name}_{total_mem_gb}GB"
    return tag, name, total_mem_gb, device_idx


def benchmark_paged_attn(
    cache: PagedQ2FP8Cache,
    layer_idx: int,
    batch_idx: int,
    q: torch.Tensor,
    page_size: int,
    delta: float,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """
    Benchmark paged attention.

    Returns:
        avg_time_ms: Average time in milliseconds
        prune_ratio: Pruning ratio
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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        output, stats = paged_attn_forward_decode(
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
    end.record()
    torch.cuda.synchronize()

    avg_time_ms = start.elapsed_time(end) / iters
    prune_ratio = stats['prune_ratio']

    return avg_time_ms, prune_ratio


def benchmark_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    iters: int,
) -> Optional[float]:
    """Benchmark FlashAttention baseline."""
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=False,
            )
        end.record()
        torch.cuda.synchronize()

        avg_time_ms = start.elapsed_time(end) / iters
        return avg_time_ms

    except ImportError:
        return None


def plot_speed_curve(
    x_lengths,
    paged_ms_list,
    flash_ms_list,
    skip_ratios,
    T_full,
    page_size,
    delta,
    layer_idx,
    gpu_label,
    out_path,
):
    """Plot performance curve aligned with ffa-q2fp8-threshold/run_attn_bench_q2_cudagraph.py style."""
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot latency
    line_paged, = ax1.plot(
        x_lengths, paged_ms_list,
        label="Paged Q2FP8",
        marker="o", markersize=2,
        color="tab:blue",
    )

    lines = [line_paged]
    labels = ["Paged Q2FP8"]

    if flash_ms_list is not None and len(flash_ms_list) > 0:
        line_flash, = ax1.plot(
            x_lengths, flash_ms_list,
            label="FlashAttn",
            marker="o", markersize=2,
            color="tab:orange",
        )
        lines.append(line_flash)
        labels.append("FlashAttn")

    ax1.set_xlabel("Sequence length (T)")
    ax1.set_ylabel("Latency per run (ms)")
    Tmax_k_str = to_k_str(T_full)
    ax1.set_title(
        f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, PageSize={page_size}, delta={delta})"
    )
    ax1.grid(True, linestyle="--", alpha=0.4)

    # GPU info box (aligned with reference style)
    if gpu_label:
        ax1.text(
            0.01, 0.99,
            f"GPU: {gpu_label}",
            transform=ax1.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Plot skip ratio on right y-axis
    ax2 = ax1.twinx()
    skip_pct = [sr * 100.0 for sr in skip_ratios]
    line_skip, = ax2.plot(
        x_lengths, skip_pct,
        label="Skip ratio (%)",
        color="tab:green", linestyle="--",
        marker="x", markersize=2,
    )
    ax2.set_ylabel("Skip ratio (%)")
    ax2.set_ylim(0, 100)

    lines.append(line_skip)
    labels.append("Skip ratio (%)")

    ax1.legend(lines, labels)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Info] Saved plot to {out_path}")


def main():
    args = parse_args()

    dtype = map_dtype(args.dtype)
    page_size = args.page_size
    delta = args.delta
    layer_idx = args.layer
    step = args.step
    warmup = args.warmup
    iters = args.iters
    max_length = args.max_length

    device = torch.device("cuda")

    # GPU info
    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    # Load real data
    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"

    print(f"[Info] Loading layer {layer_idx} from {layer_data_root}")

    # Load single layer
    gen = load_qkvh(str(layer_data_root), device=device, start_layer=layer_idx, max_length=max_length)
    data = next(gen)

    # Extract data
    # q_rope: [B=1, Hq, T, D]
    # k_rope: [B=1, Hkv, T, D]
    # v: [B=1, Hkv, T, D]
    q_rope_full = data['q_rope']
    k_rope_full = data['k_rope']
    v_full = data['v']

    B, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape

    print(f"[Info] Loaded: B={B}, Hq={Hq}, Hkv={Hkv}, T={T_full}, K={K}, V={V}")

    # Convert to expected format
    # Original: [B, Hq, T, K]
    # Expected: [B, T, Hq, K] for K/V, [B, HQ, K] for Q
    k_full = k_rope_full.transpose(1, 2).to(dtype)  # [B, T, Hkv, K]
    v_full_t = v_full.transpose(1, 2).to(dtype)      # [B, T, Hkv, V]

    # Create cache and populate
    max_pages = (T_full + page_size - 1) // page_size + 10
    cache = PagedQ2FP8Cache(
        page_size=page_size,
        max_pages=max_pages,
        max_batch_size=1,
        num_layers=1,
        use_fp8_residual=True,
        device=device,
        dtype=dtype,
    )

    print(f"[Info] Populating cache with full sequence (T={T_full})...")
    # Update cache with full K/V
    cache.update(
        k_full.squeeze(0),  # [T, Hkv, K]
        v_full_t.squeeze(0),  # [T, Hkv, V]
        layer_idx=0,
        batch_idx=0,
    )

    layer_cache = cache.get_layer(0)
    print(f"[Info] Cache populated: {layer_cache.num_pages_per_batch[0].item()} pages")

    # Generate test lengths
    lengths = list(range(step, T_full, step)) + [T_full]
    print(f"[Info] Testing {len(lengths)} lengths: {to_k_str(lengths[0])} to {to_k_str(lengths[-1])}")

    # Results storage
    x_lengths = []
    paged_ms_list = []
    flash_ms_list = [] if not args.no_flash else None
    skip_ratios = []

    print(f"\n{'='*80}")
    print("Running Benchmarks")
    print(f"{'='*80}\n")

    for L in tqdm(lengths, desc=f"Layer {layer_idx}, delta={delta}"):
        # Get query for this length (last token)
        q_rope_1 = q_rope_full[:, :, L-1:L, :].contiguous()  # [B, Hq, 1, K]
        q = q_rope_1[:, :, 0, :].unsqueeze(1).to(dtype)  # [B, 1, Hq, K]

        # Update cache sequence length for this test
        layer_cache.seq_lens[0] = L

        # Benchmark paged attention
        paged_ms, prune_ratio = benchmark_paged_attn(
            cache=cache,
            layer_idx=0,
            batch_idx=0,
            q=q,
            page_size=page_size,
            delta=delta,
            warmup=warmup,
            iters=iters,
        )

        # Benchmark FlashAttention
        flash_ms = None
        if not args.no_flash:
            k_L = k_rope_full[:, :, :L, :].contiguous()
            v_L = v_full[:, :, :L, :].contiguous()
            q_flash = q_rope_1

            flash_ms = benchmark_flash_attn(
                q=q_flash,
                k=k_L,
                v=v_L,
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
        if L == lengths[0] or L == T_full or L % (step * 4) == 0:
            speedup_str = f"{flash_ms / paged_ms:.2f}x" if flash_ms is not None else "N/A"
            flash_str = f"{flash_ms:.3f}ms" if flash_ms is not None else "N/A"
            print(f"\n  {to_k_str(L):>6}: Paged={paged_ms:>7.3f}ms, Flash={flash_str:>10}, "
                  f"Speedup={speedup_str:>6}, Prune={prune_ratio*100:>5.1f}%")

    # Save results
    output_dir = Path(__file__).parent / "plot" / "paged_real_data" / gpu_tag / f"layer_{layer_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    results = {
        "x_lengths": x_lengths,
        "paged_ms_list": paged_ms_list,
        "flash_ms_list": flash_ms_list,
        "skip_ratios": skip_ratios,
        "config": {
            "layer_idx": layer_idx,
            "T_full": T_full,
            "page_size": page_size,
            "delta": delta,
            "Hq": Hq,
            "Hkv": Hkv,
            "K": K,
            "V": V,
            "step": step,
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

    json_path = output_dir / f"results_layer{layer_idx}_page{page_size}_delta{delta}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Info] Saved results to {json_path}")

    # Plot
    if not args.no_plot:
        plot_path = output_dir / f"performance_layer{layer_idx}_page{page_size}_delta{delta}.png"
        plot_speed_curve(
            x_lengths=x_lengths,
            paged_ms_list=paged_ms_list,
            flash_ms_list=flash_ms_list,
            skip_ratios=skip_ratios,
            T_full=T_full,
            page_size=page_size,
            delta=delta,
            layer_idx=layer_idx,
            gpu_label=gpu_label,
            out_path=plot_path,
        )

    # Print summary
    print(f"\n{'='*80}")
    print("Benchmark Summary")
    print(f"{'='*80}")
    print(f"  Layer: {layer_idx}")
    print(f"  Sequence length: {T_full} ({to_k_str(T_full)})")
    print(f"  Paged Q2FP8 avg: {sum(paged_ms_list)/len(paged_ms_list):.3f} ms")
    if flash_ms_list and len(flash_ms_list) > 0:
        print(f"  FlashAttn avg: {sum(flash_ms_list)/len(flash_ms_list):.3f} ms")
        avg_speedup = sum(f/p for f, p in zip(flash_ms_list, paged_ms_list)) / len(paged_ms_list)
        print(f"  Avg speedup: {avg_speedup:.2f}x")
    print(f"  Avg prune ratio: {sum(skip_ratios)/len(skip_ratios)*100:.2f}%")
    print(f"  Peak prune ratio: {max(skip_ratios)*100:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

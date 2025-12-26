# utils/plot.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .cache import to_k_str

__all__ = ["plot_speed_curve"]


def plot_speed_curve(
    x_lengths,
    fused_ms_list,
    flash_ms_list,
    T_full,
    BS,
    SBS,
    delta,
    layer_idx,
    out_dir,
    attn_kernel_name=None,
    skip_ratios=None,
    gpu_label=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    line_fused, = ax1.plot(x_lengths, fused_ms_list, label="Triton fused", marker="o", markersize=2, color="tab:blue")
    line_flash, = ax1.plot(x_lengths, flash_ms_list, label="FlashAttn", marker="o", markersize=2, color="tab:orange")
    ax1.set_xlabel("Sequence length (T)")
    ax1.set_ylabel("Latency per run (ms)")
    Tmax_k_str = to_k_str(T_full)

    kernel_info = f" | Kernel: {attn_kernel_name}" if attn_kernel_name else ""
    ax1.set_title(
        f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta}{kernel_info})"
    )

    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = [line_fused, line_flash]
    labels = ["Triton fused", "FlashAttn"]

    if gpu_label:
        ax1.text(
            0.01,
            0.99,
            f"GPU: {gpu_label}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if skip_ratios is not None:
        ax2 = ax1.twinx()
        skip_pct = [sr * 100.0 for sr in skip_ratios]
        line_skip, = ax2.plot(
            x_lengths,
            skip_pct,
            label="Skip ratio (%)",
            color="tab:green",
            linestyle="--",
            marker="x",
            markersize=2,
        )
        ax2.set_ylabel("Skip ratio (%)")
        ax2.set_ylim(0, 100)
        lines.append(line_skip)
        labels.append("Skip ratio (%)")

    ax1.legend(lines, labels)

    if attn_kernel_name:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_{attn_kernel_name}.png"
    else:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .cache import to_k_str

__all__ = ["plot_meta_curve", "plot_speed_curve"]


def plot_meta_curve(
    x_lengths,
    update_ms_list,
    meta_ms_list,
    flash_ms_list,
    T_full,
    SBS,
    layer_idx,
    out_dir,
    kernel_name=None,
    gpu_label=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    has_meta = meta_ms_list is not None and any(x is not None for x in meta_ms_list)
    has_flash = flash_ms_list is not None and any(x is not None for x in flash_ms_list)
    update_label = "Update+Meta" if has_meta else "Update (meta precomputed)"
    line_update, = ax.plot(
        x_lengths,
        update_ms_list,
        label=update_label,
        marker="o",
        markersize=2,
        color="tab:blue",
    )
    lines = [line_update]
    labels = [update_label]
    if has_meta:
        line_meta, = ax.plot(
            x_lengths,
            meta_ms_list,
            label="Meta Only",
            marker="x",
            markersize=2,
            color="tab:green",
        )
        lines.append(line_meta)
        labels.append("Meta Only")
    if has_flash:
        line_flash, = ax.plot(
            x_lengths,
            flash_ms_list,
            label="FlashAttn",
            marker="o",
            markersize=2,
            color="tab:orange",
        )
        lines.append(line_flash)
        labels.append("FlashAttn")

    ax.set_xlabel("Sequence length (T)")
    ax.set_ylabel("Latency per run (ms)")
    Tmax_k_str = to_k_str(T_full)
    kernel_info = f" | Kernel: {kernel_name}" if kernel_name else ""
    if has_meta:
        title_prefix = "Meta Time"
    elif has_flash:
        title_prefix = "Update/Flash Time"
    else:
        title_prefix = "Update Time"
    ax.set_title(
        f"Layer {layer_idx} {title_prefix} vs Length (Tmax={Tmax_k_str}, SBS={SBS}{kernel_info})"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(lines, labels)

    if gpu_label:
        ax.text(
            0.01,
            0.99,
            f"GPU: {gpu_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if kernel_name:
        plot_path = out_dir / f"layer_{layer_idx}_meta_Tmax{Tmax_k_str}_{kernel_name}.png"
    else:
        plot_path = out_dir / f"layer_{layer_idx}_meta_Tmax{Tmax_k_str}.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def plot_speed_curve(
    x_lengths,
    paged_ms_list,
    flash_ms_list,
    T_full,
    SBS,
    delta,
    layer_idx,
    out_dir,
    kernel_name=None,
    gpu_label=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    line_paged, = ax.plot(
        x_lengths,
        paged_ms_list,
        label="Paged Q2",
        marker="o",
        markersize=2,
        color="tab:blue",
    )
    lines = [line_paged]
    labels = ["Paged Q2"]
    if flash_ms_list is not None and any(x is not None for x in flash_ms_list):
        line_flash, = ax.plot(
            x_lengths,
            flash_ms_list,
            label="FlashAttn",
            marker="o",
            markersize=2,
            color="tab:orange",
        )
        lines.append(line_flash)
        labels.append("FlashAttn")

    ax.set_xlabel("Sequence length (T)")
    ax.set_ylabel("Latency per run (ms)")
    Tmax_k_str = to_k_str(T_full)
    kernel_info = f" | Kernel: {kernel_name}" if kernel_name else ""
    ax.set_title(
        f"Layer {layer_idx} Paged Attention vs Length (Tmax={Tmax_k_str}, SBS={SBS}, delta={delta}{kernel_info})"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(lines, labels)

    if gpu_label:
        ax.text(
            0.01,
            0.99,
            f"GPU: {gpu_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if kernel_name:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_{kernel_name}.png"
    else:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path

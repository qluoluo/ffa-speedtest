#!/usr/bin/env python3
"""
可视化 prefill 和 decode 阶段的性能对比
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(json_file):
    """Load benchmark results from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_prefill_decode_comparison(results, output_dir="."):
    """Generate comprehensive comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract data
    prompt_lengths = sorted(set(r['config']['prompt_length'] for r in results))
    decode_lengths = sorted(set(r['config']['decode_length'] for r in results))

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # === 1. Prefill Time vs Prompt Length (for each decode length) ===
    ax1 = fig.add_subplot(gs[0, 0])
    for decode_len in decode_lengths:
        baseline_prefill = []
        q2fp8_prefill = []
        x_vals = []
        for prompt_len in prompt_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(prompt_len)
                baseline_prefill.append(r['baseline']['avg_prefill_ms'])
                q2fp8_prefill.append(r['q2fp8']['avg_prefill_ms'])

        if x_vals:
            ax1.plot(x_vals, baseline_prefill, 'o-', label=f'Baseline (decode={decode_len})', linewidth=2)
            ax1.plot(x_vals, q2fp8_prefill, 's--', label=f'Q2FP8 (decode={decode_len})', linewidth=2)

    ax1.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax1.set_ylabel('Prefill Time (ms)', fontweight='bold')
    ax1.set_title('Prefill Time vs Prompt Length', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')

    # === 2. Decode Time vs Decode Length (for each prompt length) ===
    ax2 = fig.add_subplot(gs[0, 1])
    for prompt_len in prompt_lengths:
        baseline_decode = []
        q2fp8_decode = []
        x_vals = []
        for decode_len in decode_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(decode_len)
                baseline_decode.append(r['baseline']['avg_decode_ms'])
                q2fp8_decode.append(r['q2fp8']['avg_decode_ms'])

        if x_vals:
            ax2.plot(x_vals, baseline_decode, 'o-', label=f'Baseline (prompt={prompt_len})', linewidth=2)
            ax2.plot(x_vals, q2fp8_decode, 's--', label=f'Q2FP8 (prompt={prompt_len})', linewidth=2)

    ax2.set_xlabel('Decode Length (tokens)', fontweight='bold')
    ax2.set_ylabel('Decode Time (ms)', fontweight='bold')
    ax2.set_title('Decode Time vs Decode Length', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')

    # === 3. Prefill Speedup Ratio ===
    ax3 = fig.add_subplot(gs[0, 2])
    for decode_len in decode_lengths:
        speedup_ratios = []
        x_vals = []
        for prompt_len in prompt_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(prompt_len)
                ratio = r['baseline']['avg_prefill_ms'] / r['q2fp8']['avg_prefill_ms']
                speedup_ratios.append(ratio)

        if x_vals:
            ax3.plot(x_vals, speedup_ratios, 'o-', label=f'decode={decode_len}', linewidth=2)

    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax3.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax3.set_ylabel('Speedup Ratio (Baseline / Q2FP8)', fontweight='bold')
    ax3.set_title('Prefill Speedup Ratio', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xscale('log')

    # === 4. Decode Speedup Ratio ===
    ax4 = fig.add_subplot(gs[1, 0])
    for prompt_len in prompt_lengths:
        speedup_ratios = []
        x_vals = []
        for decode_len in decode_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(decode_len)
                ratio = r['baseline']['avg_decode_ms'] / r['q2fp8']['avg_decode_ms']
                speedup_ratios.append(ratio)

        if x_vals:
            ax4.plot(x_vals, speedup_ratios, 'o-', label=f'prompt={prompt_len}', linewidth=2)

    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax4.set_xlabel('Decode Length (tokens)', fontweight='bold')
    ax4.set_ylabel('Speedup Ratio (Baseline / Q2FP8)', fontweight='bold')
    ax4.set_title('Decode Speedup Ratio', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale('log')

    # === 5. Per-Token Decode Time ===
    ax5 = fig.add_subplot(gs[1, 1])
    for prompt_len in prompt_lengths:
        baseline_per_token = []
        q2fp8_per_token = []
        x_vals = []
        for decode_len in decode_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(decode_len)
                baseline_per_token.append(r['baseline']['avg_per_token_ms'])
                q2fp8_per_token.append(r['q2fp8']['avg_per_token_ms'])

        if x_vals:
            ax5.plot(x_vals, baseline_per_token, 'o-', label=f'Baseline (prompt={prompt_len})', linewidth=2)
            ax5.plot(x_vals, q2fp8_per_token, 's--', label=f'Q2FP8 (prompt={prompt_len})', linewidth=2)

    ax5.set_xlabel('Decode Length (tokens)', fontweight='bold')
    ax5.set_ylabel('Per-Token Time (ms)', fontweight='bold')
    ax5.set_title('Per-Token Decode Time', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_xscale('log')

    # === 6. Memory Usage ===
    ax6 = fig.add_subplot(gs[1, 2])
    for decode_len in decode_lengths:
        baseline_mem = []
        q2fp8_mem = []
        x_vals = []
        for prompt_len in prompt_lengths:
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                x_vals.append(prompt_len)
                baseline_mem.append(r['baseline']['memory_mb'])
                q2fp8_mem.append(r['q2fp8']['memory_mb'])

        if x_vals:
            ax6.plot(x_vals, baseline_mem, 'o-', label=f'Baseline (decode={decode_len})', linewidth=2)
            ax6.plot(x_vals, q2fp8_mem, 's--', label=f'Q2FP8 (decode={decode_len})', linewidth=2)

    ax6.set_xlabel('Prompt Length (tokens)', fontweight='bold')
    ax6.set_ylabel('Peak Memory (MB)', fontweight='bold')
    ax6.set_title('Memory Usage', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.set_xscale('log')

    # === 7. Heatmap: Decode Speedup ===
    ax7 = fig.add_subplot(gs[2, 0])
    speedup_matrix = np.zeros((len(prompt_lengths), len(decode_lengths)))
    for i, prompt_len in enumerate(prompt_lengths):
        for j, decode_len in enumerate(decode_lengths):
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                ratio = r['baseline']['avg_decode_ms'] / r['q2fp8']['avg_decode_ms']
                speedup_matrix[i, j] = ratio

    im = ax7.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.5)
    ax7.set_xticks(range(len(decode_lengths)))
    ax7.set_yticks(range(len(prompt_lengths)))
    ax7.set_xticklabels(decode_lengths)
    ax7.set_yticklabels(prompt_lengths)
    ax7.set_xlabel('Decode Length', fontweight='bold')
    ax7.set_ylabel('Prompt Length', fontweight='bold')
    ax7.set_title('Decode Speedup Heatmap', fontweight='bold')

    # Add text annotations
    for i in range(len(prompt_lengths)):
        for j in range(len(decode_lengths)):
            text = ax7.text(j, i, f'{speedup_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax7, label='Speedup Ratio')

    # === 8. Heatmap: Prefill Speedup ===
    ax8 = fig.add_subplot(gs[2, 1])
    prefill_speedup_matrix = np.zeros((len(prompt_lengths), len(decode_lengths)))
    for i, prompt_len in enumerate(prompt_lengths):
        for j, decode_len in enumerate(decode_lengths):
            r = next((r for r in results if r['config']['prompt_length'] == prompt_len
                     and r['config']['decode_length'] == decode_len), None)
            if r and r['baseline'] and r['q2fp8']:
                ratio = r['baseline']['avg_prefill_ms'] / r['q2fp8']['avg_prefill_ms']
                prefill_speedup_matrix[i, j] = ratio

    im = ax8.imshow(prefill_speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.5)
    ax8.set_xticks(range(len(decode_lengths)))
    ax8.set_yticks(range(len(prompt_lengths)))
    ax8.set_xticklabels(decode_lengths)
    ax8.set_yticklabels(prompt_lengths)
    ax8.set_xlabel('Decode Length', fontweight='bold')
    ax8.set_ylabel('Prompt Length', fontweight='bold')
    ax8.set_title('Prefill Speedup Heatmap', fontweight='bold')

    # Add text annotations
    for i in range(len(prompt_lengths)):
        for j in range(len(decode_lengths)):
            text = ax8.text(j, i, f'{prefill_speedup_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax8, label='Speedup Ratio')

    # === 9. Total Time Breakdown ===
    ax9 = fig.add_subplot(gs[2, 2])
    # Pick a representative configuration
    rep_prompt = prompt_lengths[len(prompt_lengths)//2]
    rep_decode = decode_lengths[len(decode_lengths)//2]
    r = next((r for r in results if r['config']['prompt_length'] == rep_prompt
             and r['config']['decode_length'] == rep_decode), None)

    if r and r['baseline'] and r['q2fp8']:
        categories = ['Baseline', 'Q2FP8']
        prefill_times = [r['baseline']['avg_prefill_ms'], r['q2fp8']['avg_prefill_ms']]
        decode_times = [r['baseline']['avg_decode_ms'], r['q2fp8']['avg_decode_ms']]

        x = np.arange(len(categories))
        width = 0.35

        p1 = ax9.bar(x, prefill_times, width, label='Prefill', color='#3498db')
        p2 = ax9.bar(x, decode_times, width, bottom=prefill_times, label='Decode', color='#e74c3c')

        ax9.set_ylabel('Time (ms)', fontweight='bold')
        ax9.set_title(f'Time Breakdown (prompt={rep_prompt}, decode={rep_decode})', fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories)
        ax9.legend()

        # Add total time labels
        for i, (pf, dec) in enumerate(zip(prefill_times, decode_times)):
            total = pf + dec
            ax9.text(i, total, f'{total:.0f}ms', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Q2FP8-Unified vs Baseline: Prefill & Decode Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_file = output_dir / "prefill_decode_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    plt.close()


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Prefill & Decode Performance")
    print("=" * 120)

    print(f"\n{'Prompt':<10} {'Decode':<10} {'Prefill (ms)':<30} {'Decode (ms)':<30} {'Per-Token (ms)':<30}")
    print(f"{'Length':<10} {'Length':<10} {'Baseline':<10} {'Q2FP8':<10} {'Ratio':<10} {'Baseline':<10} {'Q2FP8':<10} {'Ratio':<10} {'Baseline':<10} {'Q2FP8':<10} {'Ratio':<10}")
    print("-" * 120)

    for r in results:
        if r['baseline'] and r['q2fp8']:
            prompt_len = r['config']['prompt_length']
            decode_len = r['config']['decode_length']

            baseline_pf = r['baseline']['avg_prefill_ms']
            q2fp8_pf = r['q2fp8']['avg_prefill_ms']
            pf_ratio = baseline_pf / q2fp8_pf

            baseline_dec = r['baseline']['avg_decode_ms']
            q2fp8_dec = r['q2fp8']['avg_decode_ms']
            dec_ratio = baseline_dec / q2fp8_dec

            baseline_pt = r['baseline']['avg_per_token_ms']
            q2fp8_pt = r['q2fp8']['avg_per_token_ms']
            pt_ratio = baseline_pt / q2fp8_pt

            print(f"{prompt_len:<10} {decode_len:<10} {baseline_pf:<10.1f} {q2fp8_pf:<10.1f} {pf_ratio:<10.3f} "
                  f"{baseline_dec:<10.1f} {q2fp8_dec:<10.1f} {dec_ratio:<10.3f} "
                  f"{baseline_pt:<10.2f} {q2fp8_pt:<10.2f} {pt_ratio:<10.3f}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Visualize prefill/decode benchmark results")
    parser.add_argument("--input", type=str, default="prefill_decode_benchmark.json",
                        help="Input JSON file with benchmark results")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for plots")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return

    print(f"Loading results from: {input_file}")
    results = load_results(input_file)

    print(f"Found {len(results)} benchmark configurations")

    # Print summary table
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_prefill_decode_comparison(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

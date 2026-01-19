#!/usr/bin/env python3
"""
Compare performance across different max_kept_ratio values.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_result(json_path):
    """Load benchmark result from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data

def find_result_files(base_dir, pattern="*_cudagraph_replay.json"):
    """Find all result files matching pattern."""
    base_path = Path(base_dir)
    files = list(base_path.rglob(pattern))
    return files

def extract_max_kept_ratio(data):
    """Extract max_kept_ratio from metadata."""
    return data.get('meta', {}).get('max_kept_ratio', 0.2)

def plot_comparison(results_by_ratio, output_path):
    """Plot performance comparison across different max_kept_ratio values."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by ratio
    ratios = sorted(results_by_ratio.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

    # Plot 1: CUDAGraph latency
    ax = axes[0, 0]
    for ratio, color in zip(ratios, colors):
        data = results_by_ratio[ratio]
        lengths = [l / 1024 for l in data['lengths']]  # Convert to K
        ax.plot(lengths, data['q2_cg_ms'], marker='o', label=f'ratio={ratio}', color=color, linewidth=2)
    ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('CUDAGraph Replay Latency vs max_kept_ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speedup over FlashAttention
    ax = axes[0, 1]
    for ratio, color in zip(ratios, colors):
        data = results_by_ratio[ratio]
        lengths = [l / 1024 for l in data['lengths']]
        speedups = [f / c if c > 0 else 0 for f, c in zip(data['flash_ms'], data['q2_cg_ms'])]
        ax.plot(lengths, speedups, marker='s', label=f'ratio={ratio}', color=color, linewidth=2)
    ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup over FlashAttention', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')

    # Plot 3: Latency at max length
    ax = axes[1, 0]
    max_latencies = []
    for ratio in ratios:
        data = results_by_ratio[ratio]
        max_latencies.append(data['q2_cg_ms'][-1])

    bars = ax.bar([str(r) for r in ratios], max_latencies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('max_kept_ratio', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(f'CUDAGraph Latency at Max Length ({int(data["lengths"][-1]/1024)}K)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, max_latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Speedup improvement
    ax = axes[1, 1]
    baseline_ratio = max(ratios)  # Use largest ratio as baseline
    baseline_data = results_by_ratio[baseline_ratio]

    improvements = []
    for ratio in ratios:
        data = results_by_ratio[ratio]
        # Calculate average speedup improvement
        baseline_latency = baseline_data['q2_cg_ms'][-1]
        current_latency = data['q2_cg_ms'][-1]
        improvement = (baseline_latency - current_latency) / baseline_latency * 100
        improvements.append(improvement)

    bars = ax.bar([str(r) for r in ratios], improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('max_kept_ratio', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title(f'Latency Improvement vs ratio={baseline_ratio} (at {int(data["lengths"][-1]/1024)}K)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    max_len_k = int(data['lengths'][-1] / 1024)
    print(f"\nAt maximum length ({max_len_k}K tokens):")
    print(f"{'Ratio':<10} {'Latency (ms)':<15} {'vs Flash':<12} {'vs ratio=0.2':<15}")
    print("-" * 60)

    flash_latency = data['flash_ms'][-1]
    baseline_latency = results_by_ratio[0.2]['q2_cg_ms'][-1]

    for ratio in ratios:
        data = results_by_ratio[ratio]
        latency = data['q2_cg_ms'][-1]
        speedup_flash = flash_latency / latency if latency > 0 else 0
        improvement = (baseline_latency - latency) / baseline_latency * 100
        print(f"{ratio:<10.2f} {latency:<15.3f} {speedup_flash:<12.2f}x {improvement:>+14.1f}%")

    print("\n" + "="*60)

def main():
    # Find all result files
    base_dir = Path("plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        sys.exit(1)

    # Find files with different max_kept_ratio
    result_files = find_result_files(base_dir, "*_cudagraph_replay.json")

    if not result_files:
        print(f"Error: No result files found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files")

    # Group by max_kept_ratio
    results_by_ratio = {}
    for file_path in result_files:
        try:
            data = load_result(file_path)
            ratio = extract_max_kept_ratio(data)

            # Only keep the most recent result for each ratio
            if ratio not in results_by_ratio:
                results_by_ratio[ratio] = data
                print(f"  Loaded ratio={ratio}: {file_path.name}")
        except Exception as e:
            print(f"  Warning: Failed to load {file_path}: {e}")

    if len(results_by_ratio) < 2:
        print(f"Error: Need at least 2 different ratios to compare, found {len(results_by_ratio)}")
        sys.exit(1)

    print(f"\nComparing {len(results_by_ratio)} different max_kept_ratio values: {sorted(results_by_ratio.keys())}")

    # Generate comparison plot
    output_path = base_dir / "max_kept_ratio_comparison.png"
    plot_comparison(results_by_ratio, output_path)

if __name__ == "__main__":
    main()

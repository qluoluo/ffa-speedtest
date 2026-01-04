#!/usr/bin/env python3
"""
Compare benchmark results between different GPUs (e.g., RTX 4090 vs H100).
Automatically finds the latest results for each GPU type.
"""

import json
import sys
from pathlib import Path
from typing import Dict


def find_latest_results(base_dir: Path, gpu_pattern: str):
    """Find the latest results directory for a given GPU pattern."""
    results_dirs = sorted(base_dir.glob(f"results_{gpu_pattern}_*"), reverse=True)
    if not results_dirs:
        return None
    return results_dirs[0]


def load_results(results_dir: Path):
    """Load benchmark results from a directory."""
    json_file = results_dir / "benchmark_results.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Results file not found: {json_file}")

    with open(json_file, 'r') as f:
        return json.load(f)


def print_comparison(gpu1_name: str, gpu1_data: Dict, gpu2_name: str, gpu2_data: Dict):
    """Print a formatted comparison of two GPU results."""

    print("=" * 100)
    print(f"Q2FP8 Optimization Benchmark Comparison: {gpu1_name} vs {gpu2_name}")
    print("=" * 100)
    print()

    # Configuration
    print("Configuration:")
    config = gpu1_data['config']
    print(f"  Sequence Length: {config['T']:,} tokens")
    print(f"  BS: {config['BS']}, SBS: {config['SBS']}, delta: {config['delta']}")
    print()

    # GPU Info
    print("GPU Information:")
    print(f"  {gpu1_name}: {gpu1_data['gpu']['name']}")
    print(f"  {gpu2_name}: {gpu2_data['gpu']['name']}")
    print()

    # Results table
    print("=" * 100)
    print(f"{'Kernel':<25} {gpu1_name + ' (ms)':<20} {gpu2_name + ' (ms)':<20} {'Speedup':<15} {'Winner':<10}")
    print("-" * 100)

    kernels = ['baseline', 'opt2_adaptive_bm_dot', 'opt3_fp16_obuf', 'opt4_stage2_compact', 'opt5_autotune']
    kernel_names = {
        'baseline': 'Baseline',
        'opt2_adaptive_bm_dot': 'Opt2 (Adaptive BM_DOT)',
        'opt3_fp16_obuf': 'Opt3 (FP16 o_buf)',
        'opt4_stage2_compact': 'Opt4 (Stage2 Compact)',
        'opt5_autotune': 'Opt5 (Autotune)',
    }

    for kernel in kernels:
        kernel_name = kernel_names[kernel]
        gpu1_result = gpu1_data['results'].get(kernel, {})
        gpu2_result = gpu2_data['results'].get(kernel, {})

        if 'error' in gpu1_result or 'error' in gpu2_result:
            gpu1_str = "ERROR" if 'error' in gpu1_result else f"{gpu1_result['mean_ms']:.4f}"
            gpu2_str = "ERROR" if 'error' in gpu2_result else f"{gpu2_result['mean_ms']:.4f}"
            print(f"{kernel_name:<25} {gpu1_str:<20} {gpu2_str:<20} {'-':<15} {'-':<10}")
        else:
            gpu1_time = gpu1_result['mean_ms']
            gpu2_time = gpu2_result['mean_ms']
            speedup = gpu1_time / gpu2_time

            winner = f"{gpu2_name} ✓" if speedup > 1.02 else (f"{gpu1_name} ✓" if speedup < 0.98 else "~Equal")
            gpu1_str = f"{gpu1_time:.4f} ± {gpu1_result['std_ms']:.4f}"
            gpu2_str = f"{gpu2_time:.4f} ± {gpu2_result['std_ms']:.4f}"
            speedup_str = f"{speedup:.3f}x"

            print(f"{kernel_name:<25} {gpu1_str:<20} {gpu2_str:<20} {speedup_str:<15} {winner:<10}")

    print("=" * 100)
    print()


def main():
    script_dir = Path(__file__).parent

    print("Searching for benchmark results...")
    print()

    rtx4090_dir = find_latest_results(script_dir, "RTX4090")
    h100_dir = find_latest_results(script_dir, "H100")

    if rtx4090_dir:
        print(f"Found RTX 4090 results: {rtx4090_dir.name}")
    if h100_dir:
        print(f"Found H100 results: {h100_dir.name}")
    print()

    if not rtx4090_dir or not h100_dir:
        print("Need results from both GPUs to compare.")
        if not h100_dir:
            print("\nTo run on H100:")
            print(f"  cd {script_dir}")
            print("  ./RUN_ON_H100.sh")
        sys.exit(1)

    rtx4090_data = load_results(rtx4090_dir)
    h100_data = load_results(h100_dir)

    print_comparison("RTX 4090", rtx4090_data, "H100", h100_data)


if __name__ == "__main__":
    main()

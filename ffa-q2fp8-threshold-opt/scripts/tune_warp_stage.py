#!/usr/bin/env python3
"""
Warp/Stage Grid Search for attn_q2fp8_sym_lr64_atomic_compact kernel.

This script performs a comprehensive grid search over num_warps and num_stages
parameters for all three kernel stages (threshold, stage1, stage2) to find
the optimal configuration.

Usage:
    python scripts/tune_warp_stage.py --quick      # Quick test (fewer combinations)
    python scripts/tune_warp_stage.py --full       # Full grid search
    python scripts/tune_warp_stage.py --stage s1   # Only tune stage1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product
import time

# Default search space
WARP_OPTIONS = [1, 2, 4, 8]
STAGE_OPTIONS = [2, 3, 4]

# Quick test (reduced search space)
WARP_OPTIONS_QUICK = [2, 4, 8]
STAGE_OPTIONS_QUICK = [2, 3]

# Test configuration
DEFAULT_BS = 128
DEFAULT_SBS = 128
DEFAULT_DELTA = 5.0
DEFAULT_MAX_KEPT_RATIO = 0.02
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 50
DEFAULT_STEP = 65536  # Test 4 length points: 64K, 128K, 192K, 256K


def run_benchmark(
    num_warps_th=None,
    num_stages_th=None,
    num_warps_s1=None,
    num_stages_s1=None,
    num_warps_s2=None,
    num_stages_s2=None,
    bs=DEFAULT_BS,
    sbs=DEFAULT_SBS,
    delta=DEFAULT_DELTA,
    max_kept_ratio=DEFAULT_MAX_KEPT_RATIO,
    iterations=DEFAULT_ITERATIONS,
    warmup=DEFAULT_WARMUP,
    step=DEFAULT_STEP,
):
    """Run a single benchmark with specified parameters."""

    cmd = [
        "python", "scripts/run_attn_bench_q2fp8_cudagraph.py",
        "--attn-kernel", "attn_q2fp8_sym_lr64_atomic_compact",
        "--bs", str(bs),
        "--delta", str(delta),
        "--max-kept-ratio", str(max_kept_ratio),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--step", str(step),
    ]

    if sbs is not None:
        cmd.extend(["--sbs", str(sbs)])

    # Add warp/stage parameters
    if num_warps_th is not None:
        cmd.extend(["--num-warps-th", str(num_warps_th)])
    if num_stages_th is not None:
        cmd.extend(["--num-stages-th", str(num_stages_th)])
    if num_warps_s1 is not None:
        cmd.extend(["--num-warps-s1", str(num_warps_s1)])
    if num_stages_s1 is not None:
        cmd.extend(["--num-stages-s1", str(num_stages_s1)])
    if num_warps_s2 is not None:
        cmd.extend(["--num-warps-s2", str(num_warps_s2)])
    if num_stages_s2 is not None:
        cmd.extend(["--num-stages-s2", str(num_stages_s2)])

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd[-20:])}")  # Print last 20 args
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"stderr: {e.stderr}")
        return False, None


def find_result_file(num_warps_th, num_stages_th, num_warps_s1, num_stages_s1,
                     num_warps_s2, num_stages_s2, max_kept_ratio, step, iterations):
    """Find the result JSON file for given parameters."""

    # Build the expected filename pattern
    base_dir = Path("plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph")

    # Build tags
    tags = []
    tags.append(f"mkr{max_kept_ratio}")
    tags.append(f"step{step}")
    tags.append(f"it{iterations}")

    # Add warp/stage tags
    if num_warps_th is not None:
        tags.append(f"wth{num_warps_th}")
    if num_stages_th is not None:
        tags.append(f"sth{num_stages_th}")
    if num_warps_s1 is not None:
        tags.append(f"ws1_{num_warps_s1}")
    if num_stages_s1 is not None:
        tags.append(f"ss1_{num_stages_s1}")
    if num_warps_s2 is not None:
        tags.append(f"ws2_{num_warps_s2}")
    if num_stages_s2 is not None:
        tags.append(f"ss2_{num_stages_s2}")

    # Search for matching files
    pattern = f"*{'*'.join(tags)}*.json"
    matches = list(base_dir.rglob(pattern))

    if matches:
        return matches[0]

    # Fallback: search more broadly
    for json_file in base_dir.rglob("*.json"):
        if all(tag in json_file.name for tag in tags):
            return json_file

    return None


def extract_latency(result_file):
    """Extract the 256K latency from result JSON."""
    try:
        with open(result_file) as f:
            data = json.load(f)

        # Get the last (256K) latency
        if 'q2_cg_ms' in data and len(data['q2_cg_ms']) > 0:
            return data['q2_cg_ms'][-1]

        return None
    except Exception as e:
        print(f"❌ Failed to extract latency: {e}")
        return None


def tune_stage(stage_name, warp_options, stage_options, quick=False):
    """Tune a specific stage."""

    print(f"\n{'='*70}")
    print(f"TUNING {stage_name.upper()}")
    print(f"{'='*70}")
    print(f"Warp options: {warp_options}")
    print(f"Stage options: {stage_options}")
    print()

    results = []
    total_combinations = len(warp_options) * len(stage_options)
    current = 0

    for num_warps, num_stages in product(warp_options, stage_options):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing {stage_name}: "
              f"num_warps={num_warps}, num_stages={num_stages}")

        # Set parameters based on stage
        kwargs = {
            'max_kept_ratio': DEFAULT_MAX_KEPT_RATIO,
            'iterations': DEFAULT_ITERATIONS,
            'warmup': DEFAULT_WARMUP,
            'step': DEFAULT_STEP,
        }

        if stage_name == 'threshold':
            kwargs['num_warps_th'] = num_warps
            kwargs['num_stages_th'] = num_stages
        elif stage_name == 'stage1':
            kwargs['num_warps_s1'] = num_warps
            kwargs['num_stages_s1'] = num_stages
        elif stage_name == 'stage2':
            kwargs['num_warps_s2'] = num_warps
            kwargs['num_stages_s2'] = num_stages

        # Run benchmark
        success, output = run_benchmark(**kwargs)

        if not success:
            print(f"⚠️  Skipping failed configuration")
            continue

        # Find and parse result file
        result_file = find_result_file(
            kwargs.get('num_warps_th'),
            kwargs.get('num_stages_th'),
            kwargs.get('num_warps_s1'),
            kwargs.get('num_stages_s1'),
            kwargs.get('num_warps_s2'),
            kwargs.get('num_stages_s2'),
            DEFAULT_MAX_KEPT_RATIO,
            DEFAULT_STEP,
            DEFAULT_ITERATIONS,
        )

        if result_file:
            latency = extract_latency(result_file)
            if latency:
                results.append({
                    'stage': stage_name,
                    'num_warps': num_warps,
                    'num_stages': num_stages,
                    'latency_ms': latency,
                    'result_file': str(result_file),
                })
                print(f"✅ Latency: {latency:.3f} ms")
            else:
                print(f"⚠️  Could not extract latency")
        else:
            print(f"⚠️  Could not find result file")

        # Small delay to avoid overwhelming the system
        time.sleep(1)

    return results


def tune_all_stages(quick=False):
    """Tune all three stages sequentially."""

    warp_opts = WARP_OPTIONS_QUICK if quick else WARP_OPTIONS
    stage_opts = STAGE_OPTIONS_QUICK if quick else STAGE_OPTIONS

    all_results = []

    # Tune each stage
    for stage in ['threshold', 'stage1', 'stage2']:
        stage_results = tune_stage(stage, warp_opts, stage_opts, quick)
        all_results.extend(stage_results)

    return all_results


def save_results(results, output_file):
    """Save tuning results to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


def print_summary(results):
    """Print summary of tuning results."""

    if not results:
        print("\n❌ No results to summarize")
        return

    print(f"\n{'='*70}")
    print("TUNING SUMMARY")
    print(f"{'='*70}\n")

    # Group by stage
    by_stage = {}
    for r in results:
        stage = r['stage']
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(r)

    # Print best for each stage
    for stage in ['threshold', 'stage1', 'stage2']:
        if stage not in by_stage:
            continue

        stage_results = by_stage[stage]
        best = min(stage_results, key=lambda x: x['latency_ms'])

        print(f"{stage.upper()}:")
        print(f"  Best: num_warps={best['num_warps']}, num_stages={best['num_stages']}")
        print(f"  Latency: {best['latency_ms']:.3f} ms")
        print()

    # Overall best
    best_overall = min(results, key=lambda x: x['latency_ms'])
    print(f"OVERALL BEST:")
    print(f"  Stage: {best_overall['stage']}")
    print(f"  num_warps={best_overall['num_warps']}, num_stages={best_overall['num_stages']}")
    print(f"  Latency: {best_overall['latency_ms']:.3f} ms")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Tune warp/stage parameters")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced search space")
    parser.add_argument("--full", action="store_true", help="Full grid search")
    parser.add_argument("--stage", choices=['threshold', 'stage1', 'stage2', 'all'],
                       default='all', help="Which stage to tune")
    parser.add_argument("--output", default="tuning_results/warp_stage_tuning.json",
                       help="Output JSON file")

    args = parser.parse_args()

    if args.quick:
        print("🚀 Running QUICK tuning (reduced search space)")
    elif args.full:
        print("🔍 Running FULL grid search")
    else:
        print("⚡ Running default tuning")

    # Run tuning
    if args.stage == 'all':
        results = tune_all_stages(quick=args.quick)
    else:
        warp_opts = WARP_OPTIONS_QUICK if args.quick else WARP_OPTIONS
        stage_opts = STAGE_OPTIONS_QUICK if args.quick else STAGE_OPTIONS
        results = tune_stage(args.stage, warp_opts, stage_opts, quick=args.quick)

    # Save and summarize
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()

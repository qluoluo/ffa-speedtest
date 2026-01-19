#!/usr/bin/env python3
"""
Block Size Tuning for attn_q2fp8_sym_lr64_atomic_compact kernel.

This script tunes BS (block size) and SBS (sub-block size) parameters.
Note: BK (K dimension block) is hardcoded to 64 in the kernel and cannot be changed
without modifying the source code.

Usage:
    python scripts/tune_block_sizes.py --quick      # Quick test
    python scripts/tune_block_sizes.py --full       # Full search
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product
import time

# Search space (BK is fixed at 64 in kernel)
BS_OPTIONS = [64, 128, 256]
SBS_OPTIONS = [32, 64, 128]

# Quick test (reduced)
BS_OPTIONS_QUICK = [128, 256]
SBS_OPTIONS_QUICK = [64, 128]

# Test configuration
DEFAULT_DELTA = 5.0
DEFAULT_MAX_KEPT_RATIO = 0.02
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 50
DEFAULT_STEP = 65536


def run_benchmark(
    bs,
    sbs,
    delta=DEFAULT_DELTA,
    max_kept_ratio=DEFAULT_MAX_KEPT_RATIO,
    iterations=DEFAULT_ITERATIONS,
    warmup=DEFAULT_WARMUP,
    step=DEFAULT_STEP,
    num_warps_th=None,
    num_stages_th=None,
    num_warps_s1=None,
    num_stages_s1=None,
    num_warps_s2=None,
    num_stages_s2=None,
):
    """Run a single benchmark with specified parameters."""

    cmd = [
        "python", "scripts/run_attn_bench_q2fp8_cudagraph.py",
        "--attn-kernel", "attn_q2fp8_sym_lr64_atomic_compact",
        "--bs", str(bs),
        "--sbs", str(sbs),
        "--delta", str(delta),
        "--max-kept-ratio", str(max_kept_ratio),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--step", str(step),
    ]

    # Add warp/stage parameters if provided
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
    print(f"Testing: BS={bs}, SBS={sbs}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"stderr: {e.stderr}")
        return False, None


def find_result_file(bs, sbs, max_kept_ratio, step, iterations):
    """Find the result JSON file for given parameters."""

    base_dir = Path("plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph")

    # Build tags
    tags = [
        f"bs{bs}",
        f"sbs{sbs}",
        f"mkr{max_kept_ratio}",
        f"step{step}",
        f"it{iterations}",
    ]

    # Search for matching files
    for json_file in base_dir.rglob("*.json"):
        if all(tag in json_file.name for tag in tags):
            return json_file

    return None


def extract_latency(result_file):
    """Extract the 256K latency from result JSON."""
    try:
        with open(result_file) as f:
            data = json.load(f)

        if 'q2_cg_ms' in data and len(data['q2_cg_ms']) > 0:
            return data['q2_cg_ms'][-1]

        return None
    except Exception as e:
        print(f"❌ Failed to extract latency: {e}")
        return None


def tune_block_sizes(bs_options, sbs_options, best_warp_stage=None):
    """Tune block size parameters."""

    print(f"\n{'='*70}")
    print("TUNING BLOCK SIZES")
    print(f"{'='*70}")
    print(f"BS options: {bs_options}")
    print(f"SBS options: {sbs_options}")
    print(f"Note: BK is fixed at 64 in the kernel")
    print()

    results = []
    total_combinations = len(bs_options) * len(sbs_options)
    current = 0

    for bs, sbs in product(bs_options, sbs_options):
        # Skip invalid combinations
        if sbs > bs:
            print(f"⚠️  Skipping invalid: SBS ({sbs}) > BS ({bs})")
            continue

        current += 1
        print(f"\n[{current}/{total_combinations}] Testing BS={bs}, SBS={sbs}")

        # Prepare kwargs
        kwargs = {
            'bs': bs,
            'sbs': sbs,
            'bk': bk,
            'max_kept_ratio': DEFAULT_MAX_KEPT_RATIO,
            'iterations': DEFAULT_ITERATIONS,
            'warmup': DEFAULT_WARMUP,
            'step': DEFAULT_STEP,
        }

        # Add best warp/stage config if available
        if best_warp_stage:
            kwargs.update(best_warp_stage)

        # Run benchmark
        success, output = run_benchmark(**kwargs)

        if not success:
            print(f"⚠️  Skipping failed configuration")
            continue

        # Find and parse result file
        result_file = find_result_file(
            bs, sbs, bk,
            DEFAULT_MAX_KEPT_RATIO,
            DEFAULT_STEP,
            DEFAULT_ITERATIONS,
        )

        if result_file:
            latency = extract_latency(result_file)
            if latency:
                results.append({
                    'bs': bs,
                    'sbs': sbs,
                    'bk': bk,
                    'latency_ms': latency,
                    'result_file': str(result_file),
                })
                print(f"✅ Latency: {latency:.3f} ms")
            else:
                print(f"⚠️  Could not extract latency")
        else:
            print(f"⚠️  Could not find result file")

        time.sleep(1)

    return results


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
    print("BLOCK SIZE TUNING SUMMARY")
    print(f"{'='*70}\n")

    # Sort by latency
    sorted_results = sorted(results, key=lambda x: x['latency_ms'])

    # Print top 5
    print("TOP 5 CONFIGURATIONS:\n")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i}. BS={r['bs']}, SBS={r['sbs']}, BK={r['bk']}")
        print(f"   Latency: {r['latency_ms']:.3f} ms")
        print()

    # Best configuration
    best = sorted_results[0]
    print(f"BEST CONFIGURATION:")
    print(f"  BS={best['bs']}, SBS={best['sbs']}, BK={best['bk']}")
    print(f"  Latency: {best['latency_ms']:.3f} ms")

    # Compare to baseline (BS=128, SBS=128, BK=64)
    baseline = next((r for r in results if r['bs'] == 128 and r['sbs'] == 128 and r['bk'] == 64), None)
    if baseline:
        improvement = (baseline['latency_ms'] - best['latency_ms']) / baseline['latency_ms'] * 100
        print(f"  Improvement vs baseline: {improvement:+.1f}%")

    print(f"{'='*70}\n")


def load_best_warp_stage(warp_stage_file):
    """Load best warp/stage configuration from previous tuning."""
    try:
        with open(warp_stage_file) as f:
            results = json.load(f)

        if not results:
            return None

        # Group by stage and find best for each
        best_config = {}
        by_stage = {}
        for r in results:
            stage = r['stage']
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(r)

        # Get best for each stage
        for stage, stage_results in by_stage.items():
            best = min(stage_results, key=lambda x: x['latency_ms'])
            if stage == 'threshold':
                best_config['num_warps_th'] = best['num_warps']
                best_config['num_stages_th'] = best['num_stages']
            elif stage == 'stage1':
                best_config['num_warps_s1'] = best['num_warps']
                best_config['num_stages_s1'] = best['num_stages']
            elif stage == 'stage2':
                best_config['num_warps_s2'] = best['num_warps']
                best_config['num_stages_s2'] = best['num_stages']

        print(f"✅ Loaded best warp/stage config: {best_config}")
        return best_config

    except Exception as e:
        print(f"⚠️  Could not load warp/stage config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Tune block size parameters")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced search space")
    parser.add_argument("--full", action="store_true", help="Full search")
    parser.add_argument("--warp-stage-config", type=str,
                       default="tuning_results/warp_stage_tuning.json",
                       help="JSON file with best warp/stage config from previous tuning")
    parser.add_argument("--output", default="tuning_results/block_size_tuning.json",
                       help="Output JSON file")

    args = parser.parse_args()

    if args.quick:
        print("🚀 Running QUICK tuning (reduced search space)")
        bs_opts = BS_OPTIONS_QUICK
        sbs_opts = SBS_OPTIONS_QUICK
        bk_opts = BK_OPTIONS_QUICK
    else:
        print("🔍 Running FULL block size tuning")
        bs_opts = BS_OPTIONS
        sbs_opts = SBS_OPTIONS
        bk_opts = BK_OPTIONS

    # Load best warp/stage config if available
    best_warp_stage = None
    if Path(args.warp_stage_config).exists():
        best_warp_stage = load_best_warp_stage(args.warp_stage_config)

    # Run tuning
    results = tune_block_sizes(bs_opts, sbs_opts, bk_opts, best_warp_stage)

    # Save and summarize
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()

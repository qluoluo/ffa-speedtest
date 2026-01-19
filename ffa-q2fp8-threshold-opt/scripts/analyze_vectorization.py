#!/usr/bin/env python3
"""
Analyze and compare vectorization performance results.
"""
import json
from pathlib import Path

def load_result(json_path):
    """Load benchmark result from JSON file."""
    with open(json_path) as f:
        return json.load(f)

def main():
    base_dir = Path("plot")

    # Find original kernel results
    original_pattern = "**/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/**/raw/*mkr0.02*step65536*it300*.json"
    original_files = list(base_dir.glob(original_pattern))

    # Find vectorized kernel results
    vec_pattern = "**/attn_q2fp8_sym_lr64_atomic_compact_vec_cudagraph/**/raw/*mkr0.02*step65536*it300*.json"
    vec_files = list(base_dir.glob(vec_pattern))

    if not original_files:
        print("❌ Original kernel results not found")
        return

    if not vec_files:
        print("❌ Vectorized kernel results not found")
        return

    original_data = load_result(original_files[0])
    vec_data = load_result(vec_files[0])

    print("="*70)
    print("STAGE2 VECTORIZATION PERFORMANCE COMPARISON")
    print("="*70)
    print()

    print(f"Original kernel: {original_files[0].name}")
    print(f"Vectorized kernel: {vec_files[0].name}")
    print()

    # Compare at different lengths
    print(f"{'Length':<10} {'Original (ms)':<15} {'Vectorized (ms)':<17} {'Improvement':<12}")
    print("-" * 70)

    for i, length in enumerate(original_data['lengths']):
        orig_latency = original_data['q2_cg_ms'][i]
        vec_latency = vec_data['q2_cg_ms'][i]
        improvement = (orig_latency - vec_latency) / orig_latency * 100

        length_k = length // 1024
        print(f"{length_k}K{'':<7} {orig_latency:<15.3f} {vec_latency:<17.3f} {improvement:>+11.1f}%")

    print()
    print("="*70)
    print("SUMMARY (at 256K tokens)")
    print("="*70)

    orig_final = original_data['q2_cg_ms'][-1]
    vec_final = vec_data['q2_cg_ms'][-1]
    improvement_final = (orig_final - vec_final) / orig_final * 100

    flash_final = original_data['flash_ms'][-1]
    orig_speedup = flash_final / orig_final
    vec_speedup = flash_final / vec_final

    print(f"Original:    {orig_final:.3f} ms ({orig_speedup:.2f}x vs Flash)")
    print(f"Vectorized:  {vec_final:.3f} ms ({vec_speedup:.2f}x vs Flash)")
    print(f"Improvement: {improvement_final:+.1f}%")
    print()

    if improvement_final > 0:
        print(f"✅ Vectorization provides {improvement_final:.1f}% speedup!")
    elif improvement_final < -1:
        print(f"❌ Vectorization is {-improvement_final:.1f}% slower")
    else:
        print(f"⚠️  Vectorization shows minimal difference ({improvement_final:+.1f}%)")

    print()
    print("="*70)

if __name__ == "__main__":
    main()

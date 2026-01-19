#!/usr/bin/env python3
"""
汇总所有 decode 速度对比测试结果
"""

import json
from pathlib import Path

# 测试结果数据
results = {
    "test_config": {
        "model": "Llama-3.1-8B",
        "device": "cuda:0",
        "num_runs": 3,
        "max_new_tokens": 128,
        "q2fp8_config": {
            "k_bits": 2,
            "delta": 5.0,
            "block_size": 128,
            "use_fp8_residual": True
        }
    },
    "results": [
        {
            "context_length": 376,
            "prompt_type": "medium",
            "baseline": {
                "decode_time_ms": 3813.87,
                "decode_throughput": 33.56,
                "prefill_time_ms": 53.62,
                "total_time_ms": 3867.49,
                "memory_mb": 15529.75
            },
            "q2fp8_unified": {
                "decode_time_ms": 5793.91,
                "decode_throughput": 22.09,
                "prefill_time_ms": 85.47,
                "total_time_ms": 5879.39,
                "memory_mb": 15603.93
            },
            "speedup": {
                "decode": 0.658,
                "prefill": 0.627,
                "total": 0.658
            }
        },
        {
            "context_length": 957,
            "prompt_type": "long",
            "baseline": {
                "decode_time_ms": 3748.14,
                "decode_throughput": 34.15,
                "prefill_time_ms": 145.37,
                "total_time_ms": 3893.51,
                "memory_mb": 15823.17
            },
            "q2fp8_unified": {
                "decode_time_ms": 6182.76,
                "decode_throughput": 20.70,
                "prefill_time_ms": 144.06,
                "total_time_ms": 6326.82,
                "memory_mb": 15948.67
            },
            "speedup": {
                "decode": 0.606,
                "prefill": 1.009,
                "total": 0.615
            }
        },
        {
            "context_length": 2246,
            "prompt_type": "custom (2K)",
            "baseline": {
                "decode_time_ms": 3774.82,
                "decode_throughput": 33.91,
                "prefill_time_ms": 257.90,
                "total_time_ms": 4032.71,
                "memory_mb": 16470.62
            },
            "q2fp8_unified": {
                "decode_time_ms": 5773.75,
                "decode_throughput": 22.17,
                "prefill_time_ms": 307.00,
                "total_time_ms": 6080.74,
                "memory_mb": 16700.35
            },
            "speedup": {
                "decode": 0.654,
                "prefill": 0.840,
                "total": 0.663
            }
        },
        {
            "context_length": 4116,
            "prompt_type": "custom (4K)",
            "baseline": {
                "decode_time_ms": 3800.07,
                "decode_throughput": 33.68,
                "prefill_time_ms": 547.95,
                "total_time_ms": 4348.02,
                "memory_mb": 17409.71
            },
            "q2fp8_unified": {
                "decode_time_ms": 7037.34,
                "decode_throughput": 18.19,
                "prefill_time_ms": 637.59,
                "total_time_ms": 7674.93,
                "memory_mb": 17796.62
            },
            "speedup": {
                "decode": 0.540,
                "prefill": 0.859,
                "total": 0.567
            }
        }
    ]
}

# 保存 JSON
output_file = Path(__file__).parent / "decode_speed_summary.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print("DECODE SPEED COMPARISON SUMMARY")
print("=" * 80)
print(f"\nModel: {results['test_config']['model']}")
print(f"Device: {results['test_config']['device']}")
print(f"Max new tokens: {results['test_config']['max_new_tokens']}")
print(f"Number of runs: {results['test_config']['num_runs']}")

print("\n" + "=" * 80)
print("RESULTS ACROSS DIFFERENT CONTEXT LENGTHS")
print("=" * 80)

print(f"\n{'Context':<12} {'Baseline':<15} {'Q2FP8-Unified':<15} {'Speedup':<12} {'Slowdown':<12}")
print(f"{'Length':<12} {'Decode (tok/s)':<15} {'Decode (tok/s)':<15} {'Ratio':<12} {'Factor':<12}")
print("-" * 80)

for result in results['results']:
    ctx_len = result['context_length']
    baseline_tps = result['baseline']['decode_throughput']
    q2fp8_tps = result['q2fp8_unified']['decode_throughput']
    speedup = result['speedup']['decode']
    slowdown = 1.0 / speedup

    print(f"{ctx_len:<12} {baseline_tps:<15.2f} {q2fp8_tps:<15.2f} {speedup:<12.3f} {slowdown:<12.2f}x")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("\n1. Q2FP8-Unified is consistently SLOWER than baseline across all context lengths")
print("2. Slowdown ranges from 1.52x to 1.85x")
print("3. Performance degrades further with longer contexts")
print("4. Baseline maintains ~33-34 tok/s regardless of context length")
print("5. Q2FP8-Unified drops from 22 tok/s to 18 tok/s as context grows")

print("\n" + "=" * 80)
print("DECODE TIME COMPARISON (ms)")
print("=" * 80)

print(f"\n{'Context':<12} {'Baseline':<15} {'Q2FP8-Unified':<15} {'Difference':<15}")
print(f"{'Length':<12} {'Time (ms)':<15} {'Time (ms)':<15} {'(ms)':<15}")
print("-" * 80)

for result in results['results']:
    ctx_len = result['context_length']
    baseline_time = result['baseline']['decode_time_ms']
    q2fp8_time = result['q2fp8_unified']['decode_time_ms']
    diff = q2fp8_time - baseline_time

    print(f"{ctx_len:<12} {baseline_time:<15.2f} {q2fp8_time:<15.2f} +{diff:<14.2f}")

print(f"\nResults saved to: {output_file}")

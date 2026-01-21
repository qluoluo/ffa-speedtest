import subprocess
import json
import os
import sys

# Combinations to test
# Focus on regimes where memory bandwidth is critical (High BS or High SeqLen)
configs = [
    # Baseline checks
    {"prefill_len": 4096, "bs": 1},
    {"prefill_len": 16384, "bs": 1},
    # Higher Batch Size (Bandwidth bound)
    {"prefill_len": 4096, "bs": 8},
    {"prefill_len": 4096, "bs": 16},
    {"prefill_len": 4096, "bs": 32},
    # Long Context (Bandwidth bound)
    {"prefill_len": 32768, "bs": 1},
]

results = []

print("Starting Benchmark Sweep...")

for conf in configs:
    pl = conf["prefill_len"]
    bs = conf["bs"]

    print(f"\nRunning: Prefill={pl}, BatchSize={bs}")

    cmd = [
        "python",
        "benchmark_comparison.py",
        "--prefill_len",
        str(pl),
        "--batch_size",
        str(bs),
        "--decode_len",
        "64",  # Short decode to save time, throughput stabilizes quickly
        "--output_dir",
        "sweep_results",
    ]

    try:
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Failed: {result.stderr}")
            continue

        # Parse stdout for the JSON summary path or the summary itself
        # The script prints "Results saved to: <path>"
        output_path = None
        for line in result.stdout.splitlines():
            if "Results saved to:" in line:
                output_path = line.split("Results saved to:")[1].strip()

        if output_path and os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)

            comp = data.get("comparison", {})
            speedup = comp.get("speedup", 0.0)

            res_entry = {
                "config": conf,
                "speedup": speedup,
                "flash_tps": comp.get("flash_throughput", 0.0),
                "ffa_tps": comp.get("ffa_throughput", 0.0),
            }
            results.append(res_entry)
            print(
                f"  -> Speedup: {speedup:.2f}x (Flash: {res_entry['flash_tps']:.1f}, FFA: {res_entry['ffa_tps']:.1f})"
            )
        else:
            print("  -> Could not parse results.")

    except Exception as e:
        print(f"Error: {e}")

print("\n\n=== SWEEP SUMMARY ===")
print(f"{'Prefill':<10} {'BS':<5} {'Speedup':<10} {'Flash TPS':<15} {'FFA TPS':<15}")
for r in results:
    c = r["config"]
    print(
        f"{c['prefill_len']:<10} {c['bs']:<5} {r['speedup']:<10.2f} {r['flash_tps']:<15.1f} {r['ffa_tps']:<15.1f}"
    )

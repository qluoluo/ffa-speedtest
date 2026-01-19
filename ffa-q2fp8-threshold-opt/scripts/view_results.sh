#!/bin/bash
# Quick view of optimization results

echo "=========================================="
echo "MAX_KEPT_RATIO OPTIMIZATION RESULTS"
echo "=========================================="
echo ""

# Show the comparison plot
if command -v feh &> /dev/null; then
    echo "Opening comparison plot with feh..."
    feh plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png &
elif command -v eog &> /dev/null; then
    echo "Opening comparison plot with eog..."
    eog plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png &
elif command -v display &> /dev/null; then
    echo "Opening comparison plot with ImageMagick..."
    display plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png &
else
    echo "Plot saved at: plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png"
fi

echo ""
echo "Performance Summary:"
echo "-------------------"
python3 -c "
import json
from pathlib import Path

base_dir = Path('plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/NVIDIA-GeForce-RTX-4090_48GB/delta5.0_layers1_BS128_SBS128_bsz1/raw')

ratios = [0.02, 0.05, 0.1, 0.2]
results = []

for ratio in ratios:
    if ratio == 0.2:
        pattern = '*step65536*_cudagraph_replay.json'
    else:
        pattern = f'*mkr{ratio}*_cudagraph_replay.json'

    files = list(base_dir.glob(pattern))
    if not files:
        continue

    with open(files[0]) as f:
        data = json.load(f)

    latency = data['q2_cg_ms'][-1]
    flash = data['flash_ms'][-1]
    speedup = flash / latency if latency > 0 else 0

    results.append((ratio, latency, speedup))

# Print table
print(f'{'Ratio':<10} {'Latency':<12} {'vs Flash':<12} {'Improvement':<12}')
print('-' * 50)

baseline_latency = results[-1][1]  # ratio=0.2

for ratio, latency, speedup in results:
    improvement = (baseline_latency - latency) / baseline_latency * 100
    print(f'{ratio:<10.2f} {latency:<12.3f} {speedup:<12.2f}x {improvement:>+11.1f}%')

print()
print(f'Best configuration: max_kept_ratio = {results[0][0]}')
print(f'Performance gain: {(baseline_latency - results[0][1]) / baseline_latency * 100:.1f}%')
print(f'Speedup vs Flash: {results[0][2]:.2f}x')
"

echo ""
echo "Full report: MAX_KEPT_OPTIMIZATION_REPORT.md"
echo ""

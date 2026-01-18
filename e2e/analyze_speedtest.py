import json

# Read the data
with open('../ffa-q2fp8-threshold-opt/plot/attn_q2fp8_sym_lr64_compact_cudagraph/NVIDIA-GeForce-RTX-4090_48GB/delta5.0_layers1_BS128_SBS128_bsz1_262144/raw/layer_layers_1_Tmax256k_Hq24_Hkv8_D128_Dv128_BS128_SBS128_delta5_fp16_kernelattn_q2fp8_sym_lr64_compact_step1024_it500_wu100_bsz1_cudagraph_replay.json', 'r') as f:
    data = json.load(f)

lengths = data['lengths']
q2_cg_ms = data['q2_cg_ms']
flash_ms = data['flash_ms']
skip_ratios = data['skip_ratios']

print("=" * 80)
print("Q2FP8 vs Flash Attention Performance Comparison")
print("=" * 80)
print(f"{'Length':<10} {'Q2FP8 (ms)':<12} {'Flash (ms)':<12} {'Speedup':<10} {'Skip Ratio':<12}")
print("-" * 80)

# Key data points
key_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

for i, length in enumerate(lengths):
    if length in key_lengths:
        q2_time = q2_cg_ms[i]
        flash_time = flash_ms[i]
        speedup = flash_time / q2_time
        skip_ratio = skip_ratios[i]
        print(f"{length:<10} {q2_time:<12.4f} {flash_time:<12.4f} {speedup:<10.2f}x {skip_ratio:<12.4f}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

# Find crossover point (where Q2FP8 becomes faster)
crossover_idx = None
for i in range(len(lengths)):
    if q2_cg_ms[i] < flash_ms[i]:
        crossover_idx = i
        break

if crossover_idx:
    print(f"Crossover point: {lengths[crossover_idx]} tokens")
    print(f"  Q2FP8: {q2_cg_ms[crossover_idx]:.4f} ms")
    print(f"  Flash: {flash_ms[crossover_idx]:.4f} ms")
    print(f"  Skip ratio: {skip_ratios[crossover_idx]:.4f}")

# Best speedup
best_speedup_idx = max(range(len(lengths)), key=lambda i: flash_ms[i] / q2_cg_ms[i])
print(f"\nBest speedup at {lengths[best_speedup_idx]} tokens:")
print(f"  Q2FP8: {q2_cg_ms[best_speedup_idx]:.4f} ms")
print(f"  Flash: {flash_ms[best_speedup_idx]:.4f} ms")
print(f"  Speedup: {flash_ms[best_speedup_idx] / q2_cg_ms[best_speedup_idx]:.2f}x")
print(f"  Skip ratio: {skip_ratios[best_speedup_idx]:.4f}")

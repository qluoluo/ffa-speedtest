#!/usr/bin/env python3
"""
可视化 Q2FP8-Unified vs Baseline 的 decode 速度对比
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 读取结果
results_file = Path(__file__).parent / "decode_speed_summary.json"
with open(results_file, 'r') as f:
    data = json.load(f)

# 提取数据
context_lengths = [r['context_length'] for r in data['results']]
baseline_tps = [r['baseline']['decode_throughput'] for r in data['results']]
q2fp8_tps = [r['q2fp8_unified']['decode_throughput'] for r in data['results']]
speedup_ratios = [r['speedup']['decode'] for r in data['results']]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Q2FP8-Unified vs Baseline Decode Speed Comparison\nLlama-3.1-8B, 128 tokens generation',
             fontsize=14, fontweight='bold')

# 1. Decode Throughput Comparison
ax1 = axes[0, 0]
x = np.arange(len(context_lengths))
width = 0.35
bars1 = ax1.bar(x - width/2, baseline_tps, width, label='Baseline (FA2)', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, q2fp8_tps, width, label='Q2FP8-Unified', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Context Length (tokens)', fontweight='bold')
ax1.set_ylabel('Decode Throughput (tokens/s)', fontweight='bold')
ax1.set_title('Decode Throughput Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(context_lengths)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# 2. Speedup Ratio
ax2 = axes[0, 1]
colors = ['#e74c3c' if s < 1.0 else '#2ecc71' for s in speedup_ratios]
bars = ax2.bar(context_lengths, speedup_ratios, color=colors, alpha=0.8)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')
ax2.set_xlabel('Context Length (tokens)', fontweight='bold')
ax2.set_ylabel('Speedup Ratio', fontweight='bold')
ax2.set_title('Q2FP8-Unified Speedup vs Baseline', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, ratio) in enumerate(zip(bars, speedup_ratios)):
    height = bar.get_height()
    slowdown = 1.0 / ratio
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.3f}x\n({slowdown:.2f}x slower)',
            ha='center', va='bottom', fontsize=8)

# 3. Decode Time Comparison
ax3 = axes[1, 0]
baseline_times = [r['baseline']['decode_time_ms'] for r in data['results']]
q2fp8_times = [r['q2fp8_unified']['decode_time_ms'] for r in data['results']]

x = np.arange(len(context_lengths))
bars1 = ax3.bar(x - width/2, baseline_times, width, label='Baseline (FA2)', color='#2ecc71', alpha=0.8)
bars2 = ax3.bar(x + width/2, q2fp8_times, width, label='Q2FP8-Unified', color='#e74c3c', alpha=0.8)

ax3.set_xlabel('Context Length (tokens)', fontweight='bold')
ax3.set_ylabel('Decode Time (ms)', fontweight='bold')
ax3.set_title('Decode Time Comparison (128 tokens)', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(context_lengths)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

# 4. Performance Degradation with Context Length
ax4 = axes[1, 1]
ax4.plot(context_lengths, baseline_tps, marker='o', linewidth=2, markersize=8,
         label='Baseline (FA2)', color='#2ecc71')
ax4.plot(context_lengths, q2fp8_tps, marker='s', linewidth=2, markersize=8,
         label='Q2FP8-Unified', color='#e74c3c')

ax4.set_xlabel('Context Length (tokens)', fontweight='bold')
ax4.set_ylabel('Decode Throughput (tokens/s)', fontweight='bold')
ax4.set_title('Throughput vs Context Length', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 添加数值标签
for i, (ctx, base, q2fp8) in enumerate(zip(context_lengths, baseline_tps, q2fp8_tps)):
    ax4.text(ctx, base, f'{base:.1f}', ha='center', va='bottom', fontsize=8)
    ax4.text(ctx, q2fp8, f'{q2fp8:.1f}', ha='center', va='top', fontsize=8)

plt.tight_layout()

# 保存图表
output_file = Path(__file__).parent / "decode_speed_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {output_file}")

# 显示图表
# plt.show()

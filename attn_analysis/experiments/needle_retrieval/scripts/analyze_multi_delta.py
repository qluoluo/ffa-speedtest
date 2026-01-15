#!/usr/bin/env python3
"""
针对不同 delta 值生成 Needle 检索分析图
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

# 配置
DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/layer_data")
RAW_TEXT_PATH = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/raw_text.txt")
TOKENIZER_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
OUTPUT_BASE = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/experiments/needle_retrieval/outputs")

NEEDLE_TEXT = "Hidden on Emerald Island is the legendary Magic Essence."


def find_needle_token_positions(tokenizer_path: str, text: str, needle: str):
    """找到 needle 在 token 序列中的位置"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokens = tokenizer.encode(text, add_special_tokens=True)

    char_pos = text.find(needle)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding['offset_mapping']

    needle_start_token = None
    needle_end_token = None

    for i, (start, end) in enumerate(offsets):
        if start <= char_pos < end and needle_start_token is None:
            needle_start_token = i
        if start < char_pos + len(needle) <= end:
            needle_end_token = i + 1
            break

    return needle_start_token, needle_end_token, len(tokens)


def compute_attention_scores(layer_idx: int, block_size: int, device: str = 'cuda'):
    """计算指定层的注意力分数"""
    layer_dir = DATA_DIR / f"layer_{layer_idx}"

    Q = torch.load(layer_dir / "q_rope.pt", map_location='cpu').float().to(device)
    K = torch.load(layer_dir / "k_rope.pt", map_location='cpu').float().to(device)

    q = Q[0, :, -1, :]  # [HQ, K_dim]
    K = K[0]  # [HKV, T, K_dim]

    HQ, K_dim = q.shape
    HKV, T, _ = K.shape
    G = HQ // HKV
    num_blocks = T // block_size

    K_blocks = K[:, :num_blocks*block_size, :].reshape(HKV, num_blocks, block_size, K_dim)
    scale = 1.0 / np.sqrt(K_dim)

    fp_scores = torch.zeros(HQ, num_blocks, device=device)

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]
        k_block = K_blocks[kv_h]
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

    return fp_scores.cpu().numpy()


def analyze_for_delta(delta: float, needle_start: int, needle_end: int, total_tokens: int, device: str):
    """针对特定 delta 值进行分析并生成图表"""

    output_dir = OUTPUT_BASE / f"delta_{delta}"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"分析 Delta = {delta}")
    print(f"{'='*60}")

    block_sizes = [64, 128, 256]
    layers_to_analyze = [0, 7, 14, 21, 30]

    # ========== 图1: 各Head的Needle相对分数 vs 阈值 ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_idx = 14
    block_size = 64
    scores = compute_attention_scores(layer_idx, block_size, device)
    HQ = scores.shape[0]

    needle_block = needle_start // block_size
    base_scores_per_head = np.array([max(scores[h, 0], scores[h, -1]) for h in range(HQ)])
    needle_scores_per_head = scores[:, needle_block] - base_scores_per_head
    threshold = -delta

    x = np.arange(HQ)

    ax.plot(x, needle_scores_per_head, 'go-', linewidth=2, markersize=8, label='Needle Score (relative)')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = -{delta}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    for h in range(HQ):
        if needle_scores_per_head[h] >= threshold:
            ax.scatter([h], [needle_scores_per_head[h]], color='green', s=100, zorder=5)
        else:
            ax.scatter([h], [needle_scores_per_head[h]], color='red', s=100, zorder=5, marker='x')

    kept = sum(1 for h in range(HQ) if needle_scores_per_head[h] >= threshold)

    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Needle Score - max(first, last)', fontsize=12)
    ax.set_title(f'Needle Relative Score vs Threshold (Delta={delta})\n'
                f'Layer {layer_idx}, Block Size {block_size} | '
                f'Needle Kept: {kept}/{HQ} heads ({kept/HQ*100:.0f}%)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(output_dir / "needle_vs_threshold_per_head.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: needle_vs_threshold_per_head.png")

    # ========== 图2: 不同层的Needle保留率 ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = sorted([int(p.name.split('_')[1]) for p in DATA_DIR.iterdir() if p.is_dir()])
    retention_rates = []

    for layer_idx in layers:
        scores = compute_attention_scores(layer_idx, block_size, device)
        HQ = scores.shape[0]
        needle_block = needle_start // block_size

        kept_count = 0
        for h in range(HQ):
            base = max(scores[h, 0], scores[h, -1])
            needle_score = scores[h, needle_block]
            if needle_score >= base - delta:
                kept_count += 1

        retention_rates.append(kept_count / HQ * 100)

    ax.bar(layers, retention_rates, color='steelblue', alpha=0.8)
    ax.axhline(y=np.mean(retention_rates), color='red', linestyle='--',
               label=f'Average: {np.mean(retention_rates):.1f}%')
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Needle Retention Rate (%)', fontsize=12)
    ax.set_title(f'Needle Retention Rate by Layer (Delta={delta}, Block Size={block_size})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "needle_retention_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: needle_retention_by_layer.png")

    # ========== 图3: 不同Block Size的保留率对比 ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    bs_retention = {}
    for bs in block_sizes:
        layer_retentions = []
        for layer_idx in layers:
            scores = compute_attention_scores(layer_idx, bs, device)
            HQ = scores.shape[0]
            nb = needle_start // bs

            kept = sum(1 for h in range(HQ)
                      if scores[h, nb] >= max(scores[h, 0], scores[h, -1]) - delta)
            layer_retentions.append(kept / HQ * 100)

        bs_retention[bs] = np.mean(layer_retentions)

    bars = ax.bar([str(bs) for bs in block_sizes],
                  [bs_retention[bs] for bs in block_sizes],
                  color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)

    for bar, bs in zip(bars, block_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bs_retention[bs]:.1f}%', ha='center', fontsize=11)

    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Average Needle Retention Rate (%)', fontsize=12)
    ax.set_title(f'Needle Retention vs Block Size (Delta={delta})', fontsize=13)
    ax.set_ylim(0, max(bs_retention.values()) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "needle_retention_by_blocksize.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: needle_retention_by_blocksize.png")

    # ========== 图4: 大海捞针详细柱状图 ==========
    fig, ax = plt.subplots(figsize=(18, 5))

    layer_idx = 14
    block_size = 64
    head = 13

    scores = compute_attention_scores(layer_idx, block_size, device)
    head_scores = scores[head]
    base = max(head_scores[0], head_scores[-1])
    relative_scores = head_scores - base
    threshold = -delta

    x = np.arange(len(relative_scores))
    colors = ['steelblue' if s >= threshold else 'lightgray' for s in relative_scores]

    needle_block = needle_start // block_size
    bars = ax.bar(x, relative_scores, color=colors, alpha=0.7, width=1.0, edgecolor='none')
    bars[needle_block].set_color('green')
    bars[needle_block].set_alpha(1.0)

    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'Threshold = -{delta}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    needle_rel_score = relative_scores[needle_block]
    ax.annotate(f'NEEDLE\nBlock {needle_block}\nScore: {needle_rel_score:.1f}',
               xy=(needle_block, needle_rel_score),
               xytext=(needle_block + 30, min(needle_rel_score + 3, -1)),
               fontsize=10, color='green', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    kept_count = sum(1 for s in relative_scores if s >= threshold)
    needle_status = "KEPT" if needle_rel_score >= threshold else "PRUNED"

    textstr = f'Delta: {delta}\nTotal Blocks: {len(relative_scores)}\n' \
              f'Kept: {kept_count} ({kept_count/len(relative_scores)*100:.1f}%)\n' \
              f'Needle: {needle_status}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    ax.set_xlabel('Block Index', fontsize=12)
    ax.set_ylabel('Score - max(first, last)', fontsize=12)
    ax.set_title(f'Needle in Haystack (Delta={delta}, Layer {layer_idx}, Head {head})', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-1, len(relative_scores))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "needle_haystack_detail.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: needle_haystack_detail.png")

    # ========== 保存统计报告 ==========
    report_path = output_dir / "report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Needle 检索分析报告 (Delta = {delta})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Needle: {NEEDLE_TEXT}\n")
        f.write(f"Token位置: [{needle_start}, {needle_end})\n")
        f.write(f"相对位置: {needle_start/total_tokens*100:.1f}%\n\n")

        f.write("各Block Size平均保留率:\n")
        for bs in block_sizes:
            f.write(f"  BS={bs}: {bs_retention[bs]:.1f}%\n")

        f.write(f"\n各层保留率 (BS=64):\n")
        for i, layer_idx in enumerate(layers):
            f.write(f"  Layer {layer_idx}: {retention_rates[i]:.1f}%\n")

    print(f"  保存: report.txt")

    return bs_retention


def main():
    print("=" * 60)
    print("多 Delta 值 Needle 检索分析")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    with open(RAW_TEXT_PATH, 'r') as f:
        raw_text = f.read()

    needle_start, needle_end, total_tokens = find_needle_token_positions(
        TOKENIZER_PATH, raw_text, NEEDLE_TEXT
    )
    print(f"Needle token 位置: [{needle_start}, {needle_end})")
    print(f"总 token 数: {total_tokens}")

    # 分析不同的 delta 值
    deltas = [5, 8, 10]
    all_results = {}

    for delta in deltas:
        results = analyze_for_delta(delta, needle_start, needle_end, total_tokens, device)
        all_results[delta] = results

    # ========== 汇总对比图 ==========
    summary_dir = OUTPUT_BASE / "delta_comparison"
    summary_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    block_sizes = [64, 128, 256]
    x = np.arange(len(block_sizes))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, delta in enumerate(deltas):
        values = [all_results[delta][bs] for bs in block_sizes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=f'Delta={delta}', color=colors[i], alpha=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', fontsize=9)

    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Average Needle Retention Rate (%)', fontsize=12)
    ax.set_title('Needle Retention Rate: Delta Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(summary_dir / "delta_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n汇总图保存: {summary_dir / 'delta_comparison.png'}")

    # 汇总表格
    print("\n" + "=" * 60)
    print("汇总: 各 Delta 值的平均 Needle 保留率")
    print("=" * 60)
    print(f"{'Delta':<10}", end="")
    for bs in block_sizes:
        print(f"{'BS='+str(bs):>12}", end="")
    print()
    print("-" * 46)
    for delta in deltas:
        print(f"{delta:<10}", end="")
        for bs in block_sizes:
            print(f"{all_results[delta][bs]:>11.1f}%", end="")
        print()

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

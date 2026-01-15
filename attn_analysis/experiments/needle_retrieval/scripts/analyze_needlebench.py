#!/usr/bin/env python3
"""
NeedleBench 稀疏注意力分析：分析各筛选方法能否准确保留 needle 所在的 block

分析目标：
1. 找到 needle 在 token 序列中的位置
2. 分析不同 block size 下各种筛选方法是否保留了包含 needle 的 block
3. 统计在所有层和所有 head 上的 needle block 保留率
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# 配置
DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/layer_data")
NEEDLE_JSON = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/data/NeedleBench/Length32000Depth42_origin_en_32k.json")
RAW_TEXT_PATH = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/raw_text.txt")
TOKENIZER_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/experiments/needle_retrieval/outputs/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# NeedleBench needle 文本
NEEDLE_TEXT = "Hidden on Emerald Island is the legendary Magic Essence."


def find_needle_token_positions(tokenizer_path: str, text: str, needle: str) -> Tuple[int, int, List[str]]:
    """
    找到 needle 在 token 序列中的起始和结束位置

    Returns:
        (start_token_idx, end_token_idx, all_tokens)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 编码整个文本
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # 找到 needle 在原文中的字符位置
    char_pos = text.find(needle)
    if char_pos == -1:
        raise ValueError(f"Needle not found in text: {needle[:50]}...")

    # 使用 offset_mapping 找到 token 位置
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

    if needle_start_token is None or needle_end_token is None:
        # 备用方法：遍历找到包含 needle 的 token 范围
        decoded_so_far = ""
        for i, token_id in enumerate(tokens):
            decoded_so_far = tokenizer.decode(tokens[:i+1])
            if needle in decoded_so_far and needle_start_token is None:
                # 回溯找到起始位置
                for j in range(i, -1, -1):
                    partial = tokenizer.decode(tokens[:j])
                    if needle not in partial:
                        needle_start_token = j
                        break
            if needle_start_token is not None:
                needle_end_token = i + 1
                if decoded_so_far.find(needle) + len(needle) <= len(decoded_so_far.rstrip()):
                    break

    return needle_start_token, needle_end_token, tokens


def get_needle_blocks(needle_start: int, needle_end: int, block_size: int, total_tokens: int) -> List[int]:
    """
    获取包含 needle 的所有 block 索引
    """
    start_block = needle_start // block_size
    end_block = (needle_end - 1) // block_size
    num_blocks = total_tokens // block_size

    needle_blocks = list(range(start_block, min(end_block + 1, num_blocks)))
    return needle_blocks


def analyze_layer_needle_retention(
    layer_idx: int,
    needle_blocks: List[int],
    block_size: int = 64,
    delta: float = 5.0,
    device: str = 'cuda'
) -> Dict:
    """
    分析单层中各方法是否保留了 needle 所在的 block

    Returns:
        包含各方法在各 head 上对 needle block 保留情况的字典
    """
    layer_dir = DATA_DIR / f"layer_{layer_idx}"

    Q = torch.load(layer_dir / "q_rope.pt", map_location='cpu').float().to(device)
    K = torch.load(layer_dir / "k_rope.pt", map_location='cpu').float().to(device)

    q = Q[0, :, -1, :]  # [HQ, K_dim] - 最后一个 query token
    K = K[0]  # [HKV, T, K_dim]

    HQ, K_dim = q.shape
    HKV, T, _ = K.shape
    G = HQ // HKV
    num_blocks = T // block_size

    K_blocks = K[:, :num_blocks*block_size, :].reshape(HKV, num_blocks, block_size, K_dim)
    scale = 1.0 / np.sqrt(K_dim)

    # 初始化分数矩阵
    fp_scores = torch.zeros(HQ, num_blocks, device=device)
    int2_scores = torch.zeros(HQ, num_blocks, device=device)
    int2_asym_scores = torch.zeros(HQ, num_blocks, device=device)
    int1_asym_scores = torch.zeros(HQ, num_blocks, device=device)
    sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    sample4_scores = torch.zeros(HQ, num_blocks, device=device)
    centroid_scores = torch.zeros(HQ, num_blocks, device=device)

    mid_idx = block_size // 2
    sample4_indices = [0, block_size//4, block_size//2, 3*block_size//4]

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
        k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

        # 1. FP (ground truth)
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

        # 2. INT2 对称量化
        k_max_abs = k_block.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
        k_scale = k_max_abs / 1.5
        k_q = torch.round(k_block / k_scale + 1.5).clamp(0, 3)

        for g in range(G):
            q_g = q_heads[g]
            q_scaled = q_g.unsqueeze(0).unsqueeze(0) * k_scale
            score_raw = (k_q * q_scaled).sum(dim=-1)
            zp_offset = 1.5 * q_scaled.sum(dim=-1)
            int2_s = (score_raw - zp_offset) * scale
            int2_scores[kv_h * G + g] = int2_s.max(dim=-1).values

        # 3. INT2 非对称量化 (2-bit-asym)
        k_mean = k_block.mean(dim=1, keepdim=True)
        k_std = k_block.std(dim=1, keepdim=True).clamp(min=1e-6)
        k_centered = k_block - k_mean
        k_scale_asym = k_std * 2
        k_q_asym = torch.round(k_centered / k_scale_asym + 1.5).clamp(0, 3)
        k_2bit_asym = k_mean + (k_q_asym - 1.5) * k_scale_asym
        int2_asym_s = torch.einsum('gk,nbk->gnb', q_heads, k_2bit_asym) * scale
        int2_asym_scores[kv_h * G : (kv_h + 1) * G] = int2_asym_s.max(dim=-1).values

        # 4. INT1 非对称量化 (1-bit-asym)
        k_1bit_asym = k_mean + torch.sign(k_block - k_mean) * k_std
        int1_asym_s = torch.einsum('gk,nbk->gnb', q_heads, k_1bit_asym) * scale
        int1_asym_scores[kv_h * G : (kv_h + 1) * G] = int1_asym_s.max(dim=-1).values

        # 5. Sample-1
        k_mid = k_block[:, mid_idx, :]
        s1 = torch.einsum('gk,nk->gn', q_heads, k_mid) * scale
        sample1_scores[kv_h * G : (kv_h + 1) * G] = s1

        # 6. Sample-4
        k_sampled = k_block[:, sample4_indices, :]
        s4 = torch.einsum('gk,nsk->gns', q_heads, k_sampled) * scale
        sample4_scores[kv_h * G : (kv_h + 1) * G] = s4.max(dim=-1).values

        # 7. Centroid
        k_mean_cent = k_block.mean(dim=1)
        centroid_s = torch.einsum('gk,nk->gn', q_heads, k_mean_cent) * scale
        centroid_scores[kv_h * G : (kv_h + 1) * G] = centroid_s

    # 计算阈值 (使用 FP 的 first 和 last block 分数)
    threshold = torch.maximum(fp_scores[:, 0], fp_scores[:, -1]) - delta
    threshold = threshold.unsqueeze(1)

    # 判断各方法是否保留 needle blocks
    methods = {
        'FP': fp_scores,
        '2bit-sym': int2_scores,
        '2bit-asym': int2_asym_scores,
        '1bit-asym': int1_asym_scores,
        'Sample-1': sample1_scores,
        'Sample-4': sample4_scores,
        'Centroid': centroid_scores,
    }

    results = {'layer': layer_idx}

    for name, scores in methods.items():
        prune = scores < threshold  # True = 被剪枝（丢弃）
        keep = ~prune  # True = 被保留

        # 检查每个 head 是否保留了所有 needle blocks
        needle_kept_per_head = []
        for h in range(HQ):
            # 检查该 head 是否保留了所有 needle blocks
            all_needle_kept = all(keep[h, nb].item() for nb in needle_blocks if nb < num_blocks)
            needle_kept_per_head.append(all_needle_kept)

        # 统计保留了 needle 的 head 比例
        retention_rate = sum(needle_kept_per_head) / len(needle_kept_per_head)
        results[f'{name}_needle_retention'] = retention_rate
        results[f'{name}_needle_kept_heads'] = needle_kept_per_head

    return results


def main():
    print("=" * 80)
    print("NeedleBench 稀疏注意力 Needle 检测分析")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 读取原始文本
    with open(RAW_TEXT_PATH, 'r') as f:
        raw_text = f.read()

    # 找到 needle 的 token 位置
    print(f"\n正在 tokenize 文本并定位 needle...")
    needle_start, needle_end, tokens = find_needle_token_positions(TOKENIZER_PATH, raw_text, NEEDLE_TEXT)
    total_tokens = len(tokens)

    print(f"总 token 数: {total_tokens}")
    print(f"Needle token 范围: [{needle_start}, {needle_end})")
    print(f"Needle 相对位置: {needle_start/total_tokens*100:.1f}%")

    # 获取所有层
    layers = sorted([int(p.name.split('_')[1]) for p in DATA_DIR.iterdir() if p.is_dir()])
    print(f"共 {len(layers)} 层")

    # 不同 block size 下的分析
    block_sizes = [64, 128, 256, 512]

    all_results = {}

    for bs in block_sizes:
        print(f"\n{'='*60}")
        print(f"Block Size = {bs}")
        print(f"{'='*60}")

        # 计算 needle 所在的 blocks
        needle_blocks = get_needle_blocks(needle_start, needle_end, bs, total_tokens)
        print(f"Needle 所在 block(s): {needle_blocks}")

        bs_results = []
        for layer_idx in tqdm(layers, desc=f"分析 BS={bs}"):
            result = analyze_layer_needle_retention(layer_idx, needle_blocks, bs, device=device)
            bs_results.append(result)

        all_results[bs] = bs_results

    # 生成报告和可视化
    methods = ['FP', '2bit-asym', '1bit-asym', 'Sample-4', 'Sample-1', 'Centroid', '2bit-sym']

    # 1. 总结表格
    print("\n" + "=" * 80)
    print("各方法在不同 Block Size 下的 Needle Block 平均保留率")
    print("=" * 80)

    summary_data = {}
    print(f"\n{'方法':<12}", end="")
    for bs in block_sizes:
        print(f"{'BS='+str(bs):>12}", end="")
    print()
    print("-" * (12 + 12 * len(block_sizes)))

    for method in methods:
        print(f"{method:<12}", end="")
        summary_data[method] = {}
        for bs in block_sizes:
            avg_retention = np.mean([r[f'{method}_needle_retention'] for r in all_results[bs]]) * 100
            summary_data[method][bs] = avg_retention
            print(f"{avg_retention:>11.1f}%", end="")
        print()

    # 2. 保存详细报告
    report_path = OUTPUT_DIR / "needlebench_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NeedleBench Needle Block 保留率分析\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Needle: {NEEDLE_TEXT}\n")
        f.write(f"Needle token 位置: [{needle_start}, {needle_end})\n")
        f.write(f"总 token 数: {total_tokens}\n")
        f.write(f"Needle 相对位置: {needle_start/total_tokens*100:.1f}%\n\n")

        f.write("各方法 Needle Block 平均保留率 (%):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'方法':<15}")
        for bs in block_sizes:
            f.write(f"{'BS='+str(bs):>12}")
        f.write("\n")
        f.write("-" * 80 + "\n")

        for method in methods:
            f.write(f"{method:<15}")
            for bs in block_sizes:
                f.write(f"{summary_data[method][bs]:>11.1f}%")
            f.write("\n")

        f.write("\n\n各层详细数据:\n")
        for bs in block_sizes:
            f.write(f"\n{'='*60}\n")
            f.write(f"Block Size = {bs}\n")
            needle_blocks = get_needle_blocks(needle_start, needle_end, bs, total_tokens)
            f.write(f"Needle blocks: {needle_blocks}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Layer':<8}")
            for method in methods:
                f.write(f"{method:>12}")
            f.write("\n")
            f.write("-" * (8 + 12 * len(methods)) + "\n")
            for r in all_results[bs]:
                f.write(f"{r['layer']:<8}")
                for method in methods:
                    f.write(f"{r[f'{method}_needle_retention']*100:>11.1f}%")
                f.write("\n")

    print(f"\n报告已保存到: {report_path}")

    # 3. 可视化
    # 图1: 各方法在不同 Block Size 下的 Needle 保留率
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_sizes))
    width = 0.12
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#95a5a6']

    for i, method in enumerate(methods):
        values = [summary_data[method][bs] for bs in block_sizes]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i], alpha=0.85)

        # 添加数值标签
        for bar, val in zip(bars, values):
            if val < 100:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Needle Retention Rate (%)', fontsize=12)
    ax.set_title('Needle Block Retention Rate by Method and Block Size\n(Higher is Better - 100% means all heads retain the needle)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect retention')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_retention_by_blocksize.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_retention_by_blocksize.png'}")

    # 图2: 各层的 Needle 保留率热力图 (BS=64)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bs = 64
    layer_indices = [r['layer'] for r in all_results[bs]]

    # 左图: 保留率
    retention_matrix = []
    for method in methods:
        retention_matrix.append([r[f'{method}_needle_retention'] * 100 for r in all_results[bs]])
    retention_matrix = np.array(retention_matrix)

    im1 = axes[0].imshow(retention_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods)
    axes[0].set_xticks(range(0, len(layer_indices), 4))
    axes[0].set_xticklabels([str(layer_indices[i]) for i in range(0, len(layer_indices), 4)])
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Method')
    axes[0].set_title(f'Needle Retention Rate by Layer (Block Size={bs})')
    plt.colorbar(im1, ax=axes[0], label='Retention Rate (%)')

    # 右图: 各方法在各层的曲线
    for i, method in enumerate(methods):
        retention = [r[f'{method}_needle_retention'] * 100 for r in all_results[bs]]
        axes[1].plot(layer_indices, retention, marker='o', markersize=3,
                    label=method, color=colors[i], alpha=0.8)

    axes[1].set_xlabel('Layer Index', fontsize=11)
    axes[1].set_ylabel('Needle Retention Rate (%)', fontsize=11)
    axes[1].set_title(f'Needle Retention Rate Across Layers (Block Size={bs})')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=100, color='green', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_retention_by_layer_bs64.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_retention_by_layer_bs64.png'}")

    # 图3: 不同 Block Size 的层级热力图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, bs in enumerate(block_sizes):
        retention_matrix = []
        for method in methods:
            retention_matrix.append([r[f'{method}_needle_retention'] * 100 for r in all_results[bs]])
        retention_matrix = np.array(retention_matrix)

        im = axes[idx].imshow(retention_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        axes[idx].set_yticks(range(len(methods)))
        axes[idx].set_yticklabels(methods)
        axes[idx].set_xticks(range(0, len(layer_indices), 4))
        axes[idx].set_xticklabels([str(layer_indices[i]) for i in range(0, len(layer_indices), 4)])
        axes[idx].set_xlabel('Layer Index')
        axes[idx].set_ylabel('Method')
        axes[idx].set_title(f'Block Size = {bs}')
        plt.colorbar(im, ax=axes[idx], label='%')

    fig.suptitle('Needle Block Retention Rate Heatmaps', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_retention_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_retention_heatmaps.png'}")

    # 图4: Block Size 变化对各方法的影响（折线图）
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        values = [summary_data[method][bs] for bs in block_sizes]
        ax.plot(block_sizes, values, marker='o', markersize=8, linewidth=2,
               label=method, color=colors[i])

    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Average Needle Retention Rate (%)', fontsize=12)
    ax.set_title('Impact of Block Size on Needle Retention', fontsize=13)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xscale('log', base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_retention_vs_blocksize.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_retention_vs_blocksize.png'}")

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

    return summary_data, all_results


if __name__ == "__main__":
    main()

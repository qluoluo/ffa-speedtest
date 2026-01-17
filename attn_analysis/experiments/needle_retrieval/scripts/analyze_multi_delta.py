#!/usr/bin/env python3
"""
针对不同 delta 值，对比不同筛选方法的 Needle 保留率

分析目标：
1. 对每个 delta 值，计算各种方法（FP、2bit-sym、2bit-asym、Sample-4 等）的保留率
2. 生成对比图表，展示不同方法在不同 delta 下的表现

用法:
    python analyze_multi_delta.py [--data-dir DATA_DIR] [--tokenizer TOKENIZER] [--output-name NAME]
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# 默认配置
DEFAULT_DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0")
DEFAULT_TOKENIZER = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
OUTPUT_BASE = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/experiments/needle_retrieval/outputs")

DEFAULT_NEEDLE = "Hidden on Emerald Island is the legendary Magic Essence."

# 方法列表和颜色
METHODS = ['FP', '2bit-sym', '2bit-asym', '1bit-asym', 'Sample-1', 'Sample-4', 'Centroid']
METHOD_COLORS = {
    'FP': '#2ecc71',
    '2bit-sym': '#3498db',
    '2bit-asym': '#e74c3c',
    '1bit-asym': '#9b59b6',
    'Sample-1': '#f39c12',
    'Sample-4': '#1abc9c',
    'Centroid': '#95a5a6',
}


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


def compute_all_method_scores(layer_idx: int, block_size: int, data_dir: Path, device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    计算指定层所有方法的注意力分数

    Returns:
        Dict[method_name, scores]: 每个方法的分数矩阵 [HQ, num_blocks]
    """
    layer_dir = data_dir / f"layer_{layer_idx}"

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
    int2_sym_scores = torch.zeros(HQ, num_blocks, device=device)
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

        # 2. INT2 对称量化 (2bit-sym)
        k_max_abs = k_block.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
        k_scale = k_max_abs / 1.5
        k_q = torch.round(k_block / k_scale + 1.5).clamp(0, 3)

        for g in range(G):
            q_g = q_heads[g]
            q_scaled = q_g.unsqueeze(0).unsqueeze(0) * k_scale
            score_raw = (k_q * q_scaled).sum(dim=-1)
            zp_offset = 1.5 * q_scaled.sum(dim=-1)
            int2_s = (score_raw - zp_offset) * scale
            int2_sym_scores[kv_h * G + g] = int2_s.max(dim=-1).values

        # 3. INT2 非对称量化 (2bit-asym)
        k_mean = k_block.mean(dim=1, keepdim=True)
        k_std = k_block.std(dim=1, keepdim=True).clamp(min=1e-6)
        k_centered = k_block - k_mean
        k_scale_asym = k_std * 2
        k_q_asym = torch.round(k_centered / k_scale_asym + 1.5).clamp(0, 3)
        k_2bit_asym = k_mean + (k_q_asym - 1.5) * k_scale_asym
        int2_asym_s = torch.einsum('gk,nbk->gnb', q_heads, k_2bit_asym) * scale
        int2_asym_scores[kv_h * G : (kv_h + 1) * G] = int2_asym_s.max(dim=-1).values

        # 4. INT1 非对称量化 (1bit-asym)
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

    return {
        'FP': fp_scores.cpu().numpy(),
        '2bit-sym': int2_sym_scores.cpu().numpy(),
        '2bit-asym': int2_asym_scores.cpu().numpy(),
        '1bit-asym': int1_asym_scores.cpu().numpy(),
        'Sample-1': sample1_scores.cpu().numpy(),
        'Sample-4': sample4_scores.cpu().numpy(),
        'Centroid': centroid_scores.cpu().numpy(),
    }


def compute_needle_retention(
    all_scores: Dict[str, np.ndarray],
    needle_blocks: List[int],
    delta: float
) -> Dict[str, float]:
    """
    计算各方法在给定 delta 下的 needle 保留率

    阈值计算: threshold = max(score[first], score[last]) - delta
    保留条件: 所有 needle block 的分数 >= threshold
    """
    results = {}

    # 使用 FP 分数计算阈值 (这是标准做法)
    fp_scores = all_scores['FP']
    HQ, num_blocks = fp_scores.shape

    # 计算每个 head 的阈值
    thresholds = np.array([max(fp_scores[h, 0], fp_scores[h, -1]) - delta for h in range(HQ)])

    for method, scores in all_scores.items():
        kept_count = 0
        for h in range(HQ):
            # 检查该 head 是否保留了所有 needle blocks
            all_kept = all(scores[h, nb] >= thresholds[h] for nb in needle_blocks if nb < num_blocks)
            if all_kept:
                kept_count += 1

        results[method] = kept_count / HQ * 100

    return results


def analyze_multi_delta(
    deltas: List[float],
    needle_start: int,
    needle_end: int,
    total_tokens: int,
    block_size: int,
    data_dir: Path,
    device: str
) -> Dict[float, Dict[str, float]]:
    """
    对多个 delta 值分析各方法的保留率
    """
    layers = sorted([int(p.name.split('_')[1]) for p in data_dir.iterdir() if p.is_dir()])
    needle_blocks = [needle_start // block_size]
    if (needle_end - 1) // block_size != needle_blocks[0]:
        needle_blocks.append((needle_end - 1) // block_size)

    print(f"Needle 所在 block(s): {needle_blocks}")

    # 收集所有层的分数
    print(f"\n计算所有层的分数 (Block Size = {block_size})...")
    all_layer_scores = []
    for layer_idx in tqdm(layers, desc="加载层数据"):
        scores = compute_all_method_scores(layer_idx, block_size, data_dir, device)
        all_layer_scores.append(scores)

    # 对每个 delta 计算保留率
    results = {}
    for delta in deltas:
        method_retentions = {method: [] for method in METHODS}

        for layer_scores in all_layer_scores:
            layer_retention = compute_needle_retention(layer_scores, needle_blocks, delta)
            for method in METHODS:
                method_retentions[method].append(layer_retention[method])

        # 计算平均保留率
        results[delta] = {method: np.mean(method_retentions[method]) for method in METHODS}

    return results


def main():
    parser = argparse.ArgumentParser(description='多 Delta 值 + 多方法 Needle 保留率对比分析')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='数据目录路径 (包含 layer_data 和 raw_text.txt)')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER,
                        help='Tokenizer 路径')
    parser.add_argument('--output-name', type=str, default=None,
                        help='输出目录名称 (默认根据数据目录自动生成)')
    parser.add_argument('--needle', type=str, default=DEFAULT_NEEDLE,
                        help='Needle 文本内容')
    args = parser.parse_args()

    data_base = Path(args.data_dir)
    data_dir = data_base / "layer_data"
    raw_text_path = data_base / "raw_text.txt"
    tokenizer_path = args.tokenizer
    needle_text = args.needle

    # 自动生成输出目录名称
    if args.output_name:
        output_name = args.output_name
    else:
        # 从路径提取模型名和数据集名
        model_name = data_base.parent.name  # e.g., Llama-3.1-8B
        dataset_name = data_base.name  # e.g., needlebench_Length32000Depth42_origin_en_32k_0
        output_name = f"{model_name}_{dataset_name}"

    output_dir = OUTPUT_BASE / "delta_method_comparison" / output_name
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("多 Delta 值 + 多方法 Needle 保留率对比分析")
    print("=" * 70)
    print(f"数据目录: {data_base}")
    print(f"输出目录: {output_dir}")
    print(f"Needle: {needle_text}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    with open(raw_text_path, 'r') as f:
        raw_text = f.read()

    needle_start, needle_end, total_tokens = find_needle_token_positions(
        tokenizer_path, raw_text, needle_text
    )
    print(f"Needle token 位置: [{needle_start}, {needle_end})")
    print(f"总 token 数: {total_tokens}")
    print(f"Needle 相对位置: {needle_start/total_tokens*100:.1f}%")

    # 分析参数
    deltas = [3.0, 5.0, 8.0, 10.0, 15.0]
    block_sizes = [64, 128, 256]

    # 对每个 block size 进行分析
    all_results = {}
    for bs in block_sizes:
        print(f"\n{'='*60}")
        print(f"Block Size = {bs}")
        print(f"{'='*60}")

        results = analyze_multi_delta(deltas, needle_start, needle_end, total_tokens, bs, data_dir, device)
        all_results[bs] = results

    # ========== 图1: 各方法在不同 delta 下的保留率 (每个 block size 一张图) ==========
    for bs in block_sizes:
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(deltas))
        width = 0.12

        for i, method in enumerate(METHODS):
            values = [all_results[bs][delta][method] for delta in deltas]
            offset = (i - len(METHODS)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method,
                         color=METHOD_COLORS[method], alpha=0.85)

        ax.set_xlabel('Delta', fontsize=12)
        ax.set_ylabel('Needle Retention Rate (%)', fontsize=12)
        ax.set_title(f'Needle Retention Rate by Method and Delta\n(Block Size = {bs})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in deltas])
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"method_comparison_bs{bs}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: method_comparison_bs{bs}.png")

    # ========== 图2: 各方法随 delta 变化的折线图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, bs in enumerate(block_sizes):
        ax = axes[idx]

        for method in METHODS:
            values = [all_results[bs][delta][method] for delta in deltas]
            ax.plot(deltas, values, marker='o', markersize=8, linewidth=2,
                   label=method, color=METHOD_COLORS[method])

        ax.set_xlabel('Delta', fontsize=11)
        ax.set_ylabel('Needle Retention Rate (%)', fontsize=11)
        ax.set_title(f'Block Size = {bs}', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)

    fig.suptitle('Needle Retention Rate vs Delta for Different Methods', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "retention_vs_delta_all_bs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: retention_vs_delta_all_bs.png")

    # ========== 图3: 热力图 - 方法 x Delta ==========
    for bs in block_sizes:
        fig, ax = plt.subplots(figsize=(10, 6))

        data = np.array([[all_results[bs][delta][method] for delta in deltas] for method in METHODS])

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(deltas)))
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_xticklabels([str(d) for d in deltas])
        ax.set_yticklabels(METHODS)
        ax.set_xlabel('Delta', fontsize=12)
        ax.set_ylabel('Method', fontsize=12)
        ax.set_title(f'Needle Retention Rate Heatmap (Block Size = {bs})', fontsize=13)

        # 添加数值标注
        for i in range(len(METHODS)):
            for j in range(len(deltas)):
                text = ax.text(j, i, f'{data[i, j]:.0f}%',
                              ha='center', va='center', fontsize=9,
                              color='white' if data[i, j] < 50 else 'black')

        plt.colorbar(im, label='Retention Rate (%)')
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_bs{bs}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: heatmap_bs{bs}.png")

    # ========== 汇总报告 ==========
    print("\n" + "=" * 70)
    print("汇总: 各方法在不同 Delta 下的 Needle 平均保留率")
    print("=" * 70)

    for bs in block_sizes:
        print(f"\n【Block Size = {bs}】")
        print(f"{'Method':<12}", end="")
        for delta in deltas:
            print(f"{'d='+str(delta):>10}", end="")
        print()
        print("-" * (12 + 10 * len(deltas)))

        for method in METHODS:
            print(f"{method:<12}", end="")
            for delta in deltas:
                print(f"{all_results[bs][delta][method]:>9.1f}%", end="")
            print()

    # 保存详细报告
    report_path = output_dir / "summary_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("多 Delta 值 + 多方法 Needle 保留率对比分析报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"数据目录: {data_base}\n")
        f.write(f"Tokenizer: {tokenizer_path}\n\n")

        f.write(f"Needle: {needle_text}\n")
        f.write(f"Token 位置: [{needle_start}, {needle_end})\n")
        f.write(f"相对位置: {needle_start/total_tokens*100:.1f}%\n\n")

        f.write("分析参数:\n")
        f.write(f"  - Delta 值: {deltas}\n")
        f.write(f"  - Block Sizes: {block_sizes}\n\n")

        for bs in block_sizes:
            f.write(f"\n{'='*60}\n")
            f.write(f"Block Size = {bs}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"{'Method':<12}")
            for delta in deltas:
                f.write(f"{'d='+str(delta):>10}")
            f.write("\n")
            f.write("-" * (12 + 10 * len(deltas)) + "\n")

            for method in METHODS:
                f.write(f"{method:<12}")
                for delta in deltas:
                    f.write(f"{all_results[bs][delta][method]:>9.1f}%")
                f.write("\n")

        f.write("\n\n关键发现:\n")
        f.write("-" * 40 + "\n")

        # 找出各 delta 下最佳方法
        for bs in block_sizes:
            f.write(f"\nBlock Size = {bs}:\n")
            for delta in deltas:
                best_method = max(METHODS, key=lambda m: all_results[bs][delta][m])
                best_rate = all_results[bs][delta][best_method]
                f.write(f"  Delta={delta}: 最佳方法 {best_method} ({best_rate:.1f}%)\n")

    print(f"\n报告已保存: {report_path}")
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

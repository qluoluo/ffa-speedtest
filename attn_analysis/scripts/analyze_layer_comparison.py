#!/usr/bin/env python3
"""
分析各方法在不同层的一致率
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/outputs/output_layer_comparison")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def analyze_layer(layer_idx: int, block_size: int = 64, delta: float = 5.0, device: str = 'cuda') -> Dict:
    """分析单层的各方法一致率"""
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

    # 初始化分数
    fp_scores = torch.zeros(HQ, num_blocks, device=device)
    int2_scores = torch.zeros(HQ, num_blocks, device=device)
    sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    sample4_scores = torch.zeros(HQ, num_blocks, device=device)
    centroid_scores = torch.zeros(HQ, num_blocks, device=device)
    minmax_scores = torch.zeros(HQ, num_blocks, device=device)
    int1_scores = torch.zeros(HQ, num_blocks, device=device)  # 1-bit量化

    mid_idx = block_size // 2
    sample4_indices = [0, block_size//4, block_size//2, 3*block_size//4]

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
        k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

        # 1. FP (ground truth)
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

        # 2. INT2
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

        # 3. Sample-1
        k_mid = k_block[:, mid_idx, :]
        s1 = torch.einsum('gk,nk->gn', q_heads, k_mid) * scale
        sample1_scores[kv_h * G : (kv_h + 1) * G] = s1

        # 4. Sample-4
        k_sampled = k_block[:, sample4_indices, :]
        s4 = torch.einsum('gk,nsk->gns', q_heads, k_sampled) * scale
        sample4_scores[kv_h * G : (kv_h + 1) * G] = s4.max(dim=-1).values

        # 5. Centroid
        k_mean = k_block.mean(dim=1)
        centroid_s = torch.einsum('gk,nk->gn', q_heads, k_mean) * scale
        centroid_scores[kv_h * G : (kv_h + 1) * G] = centroid_s

        # 6. MinMax
        k_min = k_block.min(dim=1).values
        k_max = k_block.max(dim=1).values
        for g in range(G):
            q_g = q_heads[g]
            k_opt = torch.where(q_g.unsqueeze(0) > 0, k_max, k_min)
            mm_s = (q_g.unsqueeze(0) * k_opt).sum(dim=-1) * scale
            minmax_scores[kv_h * G + g] = mm_s

        # 7. 1-bit量化: k_1bit = k_mean + sign(k - k_mean) * k_std
        k_mean = k_block.mean(dim=1, keepdim=True)  # [num_blocks, 1, K_dim]
        k_std = k_block.std(dim=1, keepdim=True)    # [num_blocks, 1, K_dim]
        k_1bit = k_mean + torch.sign(k_block - k_mean) * k_std  # [num_blocks, block_size, K_dim]
        int1_s = torch.einsum('gk,nbk->gnb', q_heads, k_1bit) * scale
        int1_scores[kv_h * G : (kv_h + 1) * G] = int1_s.max(dim=-1).values

    # 计算阈值
    threshold = torch.maximum(fp_scores[:, 0], fp_scores[:, -1]) - delta
    threshold = threshold.unsqueeze(1)

    fp_prune = fp_scores < threshold
    fp_prune_rate = fp_prune.float().mean().item()

    results = {
        'layer': layer_idx,
        'fp_prune_rate': fp_prune_rate,
    }

    methods = {
        'INT2': int2_scores,
        'Sample-1': sample1_scores,
        'Sample-4': sample4_scores,
        'Centroid': centroid_scores,
        'MinMax': minmax_scores,
        '1-bit': int1_scores,
    }

    for name, scores in methods.items():
        prune = scores < threshold
        agreement = (prune == fp_prune).float().mean().item()
        false_pos = ((prune) & (~fp_prune)).float().mean().item()
        false_neg = ((~prune) & (fp_prune)).float().mean().item()
        prune_rate = prune.float().mean().item()

        results[f'{name}_agreement'] = agreement
        results[f'{name}_false_pos'] = false_pos
        results[f'{name}_false_neg'] = false_neg
        results[f'{name}_prune_rate'] = prune_rate

    return results


def main():
    print("分析各层的方法一致率...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size = 64

    # 获取所有层
    layers = sorted([int(p.name.split('_')[1]) for p in DATA_DIR.iterdir() if p.is_dir()])

    all_results = []
    for layer_idx in tqdm(layers, desc="分析各层"):
        result = analyze_layer(layer_idx, block_size, device=device)
        all_results.append(result)

    methods = ['INT2', 'Sample-1', 'Sample-4', 'Centroid', '1-bit', 'MinMax']

    # 计算平均值
    avg_results = {}
    for method in methods:
        avg_results[f'{method}_agreement'] = np.mean([r[f'{method}_agreement'] for r in all_results])
        avg_results[f'{method}_false_pos'] = np.mean([r[f'{method}_false_pos'] for r in all_results])
        avg_results[f'{method}_false_neg'] = np.mean([r[f'{method}_false_neg'] for r in all_results])
    avg_results['fp_prune_rate'] = np.mean([r['fp_prune_rate'] for r in all_results])

    # Layer 14 结果
    layer14_result = [r for r in all_results if r['layer'] == 14][0]

    # 生成报告
    print("\n" + "=" * 80)
    print("各方法统计 (Block Size = 64)")
    print("=" * 80)
    print(f"\n{'方法':<12} {'平均一致率':>12} {'L14一致率':>12} {'平均误剪率':>12} {'L14误剪率':>12}")
    print("-" * 60)
    for method in methods:
        avg_agr = avg_results[f'{method}_agreement'] * 100
        l14_agr = layer14_result[f'{method}_agreement'] * 100
        avg_fp = avg_results[f'{method}_false_pos'] * 100
        l14_fp = layer14_result[f'{method}_false_pos'] * 100
        print(f"{method:<12} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}% {l14_fp:>11.2f}%")

    # 保存报告
    report_path = OUTPUT_DIR / "layer_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("各方法在不同层的一致率分析 (Block Size = 64)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"FP平均剪枝率: {avg_results['fp_prune_rate']*100:.1f}%\n")
        f.write(f"FP Layer14剪枝率: {layer14_result['fp_prune_rate']*100:.1f}%\n\n")

        f.write(f"{'方法':<12} {'平均一致率':>12} {'L14一致率':>12} {'平均误剪率':>12} {'L14误剪率':>12}\n")
        f.write("-" * 60 + "\n")
        for method in methods:
            avg_agr = avg_results[f'{method}_agreement'] * 100
            l14_agr = layer14_result[f'{method}_agreement'] * 100
            avg_fp = avg_results[f'{method}_false_pos'] * 100
            l14_fp = layer14_result[f'{method}_false_pos'] * 100
            f.write(f"{method:<12} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}% {l14_fp:>11.2f}%\n")

        f.write("\n\n各层详细数据:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Layer':<8}")
        for method in methods:
            f.write(f"{method:>12}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            f.write(f"{r['layer']:<8}")
            for method in methods:
                f.write(f"{r[f'{method}_agreement']*100:>11.1f}%")
            f.write("\n")

    print(f"\n报告已保存到: {report_path}")

    # 生成图表
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_indices = [r['layer'] for r in all_results]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for i, method in enumerate(methods):
        agreements = [r[f'{method}_agreement'] * 100 for r in all_results]
        ax.plot(layer_indices, agreements,
                color=colors[i], marker=markers[i], markersize=6,
                linewidth=2, label=method, alpha=0.8)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Agreement Rate vs Layer for Different Methods (Block Size = 64)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 102)
    ax.set_xticks(layer_indices)

    # 标注Layer 0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Layer 0\n(low FP prune)', xy=(0, 55), fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_agreement_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存到: {OUTPUT_DIR / 'layer_agreement_comparison.png'}")

    # 返回统计数据用于更新文档
    return avg_results, layer14_result, methods


if __name__ == "__main__":
    main()

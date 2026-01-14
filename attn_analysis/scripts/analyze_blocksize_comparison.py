#!/usr/bin/env python3
"""
分析各方法在不同 Block Size 下的一致率
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/outputs/output_blocksize_comparison")
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

    if num_blocks == 0:
        return None

    K_blocks = K[:, :num_blocks*block_size, :].reshape(HKV, num_blocks, block_size, K_dim)
    scale = 1.0 / np.sqrt(K_dim)

    # 初始化分数
    fp_scores = torch.zeros(HQ, num_blocks, device=device)
    int2_scores = torch.zeros(HQ, num_blocks, device=device)
    sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    sample4_scores = torch.zeros(HQ, num_blocks, device=device)
    centroid_scores = torch.zeros(HQ, num_blocks, device=device)
    minmax_scores = torch.zeros(HQ, num_blocks, device=device)
    int1_scores = torch.zeros(HQ, num_blocks, device=device)

    mid_idx = block_size // 2
    sample4_indices = [0, block_size//4, block_size//2, 3*block_size//4]

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]
        k_block = K_blocks[kv_h]

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

        # 7. 1-bit量化
        k_mean = k_block.mean(dim=1, keepdim=True)
        k_std = k_block.std(dim=1, keepdim=True)
        k_1bit = k_mean + torch.sign(k_block - k_mean) * k_std
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


def analyze_block_size(block_size: int, device: str = 'cuda'):
    """分析指定 block size 的所有层"""
    layers = sorted([int(p.name.split('_')[1]) for p in DATA_DIR.iterdir() if p.is_dir()])

    all_results = []
    for layer_idx in tqdm(layers, desc=f"BS={block_size}"):
        result = analyze_layer(layer_idx, block_size, device=device)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return None, None

    methods = ['INT2', 'Sample-1', 'Sample-4', 'Centroid', '1-bit', 'MinMax']

    # 计算平均值
    avg_results = {}
    for method in methods:
        avg_results[f'{method}_agreement'] = np.mean([r[f'{method}_agreement'] for r in all_results])
        avg_results[f'{method}_false_pos'] = np.mean([r[f'{method}_false_pos'] for r in all_results])
        avg_results[f'{method}_false_neg'] = np.mean([r[f'{method}_false_neg'] for r in all_results])
    avg_results['fp_prune_rate'] = np.mean([r['fp_prune_rate'] for r in all_results])

    # Layer 14 结果
    layer14_results = [r for r in all_results if r['layer'] == 14]
    layer14_result = layer14_results[0] if layer14_results else None

    return avg_results, layer14_result


def main():
    print("分析不同 Block Size 下的方法一致率...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_sizes = [64, 128, 256, 512]
    methods = ['INT2', 'Sample-1', 'Sample-4', 'Centroid', '1-bit', 'MinMax']

    all_bs_results = {}

    for bs in block_sizes:
        avg_results, layer14_result = analyze_block_size(bs, device=device)
        all_bs_results[bs] = {
            'avg': avg_results,
            'l14': layer14_result
        }

    # 生成报告
    print("\n" + "=" * 100)
    print("各 Block Size 下的方法一致率对比")
    print("=" * 100)

    # 按方法输出
    for method in methods:
        print(f"\n### {method}")
        print(f"{'Block Size':<12} {'平均一致率':>12} {'L14一致率':>12} {'平均误剪率':>12} {'L14误剪率':>12}")
        print("-" * 60)
        for bs in block_sizes:
            avg = all_bs_results[bs]['avg']
            l14 = all_bs_results[bs]['l14']
            avg_agr = avg[f'{method}_agreement'] * 100
            l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
            avg_fp = avg[f'{method}_false_pos'] * 100
            l14_fp = l14[f'{method}_false_pos'] * 100 if l14 else 0
            print(f"{bs:<12} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}% {l14_fp:>11.2f}%")

    # 保存报告
    report_path = OUTPUT_DIR / "blocksize_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("各 Block Size 下的方法一致率对比\n")
        f.write("=" * 100 + "\n\n")

        # 按 Block Size 输出表格 (用于 CONCLUSIONS.md)
        for bs in block_sizes:
            avg = all_bs_results[bs]['avg']
            l14 = all_bs_results[bs]['l14']

            f.write(f"\n### Block Size = {bs}\n")
            f.write(f"FP平均剪枝率: {avg['fp_prune_rate']*100:.1f}%\n\n")
            f.write(f"| 方法 | 平均一致率 | L14一致率 | 平均误剪率 | L14误剪率 |\n")
            f.write(f"|------|-----------|----------|-----------|----------|\n")
            for method in methods:
                avg_agr = avg[f'{method}_agreement'] * 100
                l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
                avg_fp = avg[f'{method}_false_pos'] * 100
                l14_fp = l14[f'{method}_false_pos'] * 100 if l14 else 0
                f.write(f"| {method} | {avg_agr:.2f}% | {l14_agr:.2f}% | {avg_fp:.2f}% | {l14_fp:.2f}% |\n")

        # 按方法输出详细数据
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("按方法分组的详细数据\n")
        f.write("=" * 100 + "\n")

        for method in methods:
            f.write(f"\n### {method}\n")
            f.write(f"| Block Size | 平均一致率 | L14一致率 | 平均误剪率 | L14误剪率 |\n")
            f.write(f"|------------|-----------|----------|-----------|----------|\n")
            for bs in block_sizes:
                avg = all_bs_results[bs]['avg']
                l14 = all_bs_results[bs]['l14']
                avg_agr = avg[f'{method}_agreement'] * 100
                l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
                avg_fp = avg[f'{method}_false_pos'] * 100
                l14_fp = l14[f'{method}_false_pos'] * 100 if l14 else 0
                f.write(f"| {bs} | {avg_agr:.2f}% | {l14_agr:.2f}% | {avg_fp:.2f}% | {l14_fp:.2f}% |\n")

    print(f"\n报告已保存到: {report_path}")

    return all_bs_results


if __name__ == "__main__":
    main()

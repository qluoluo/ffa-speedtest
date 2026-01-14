#!/usr/bin/env python3
"""
分析量化+采样组合方法的效果
- 量化减少存储/带宽
- 采样减少计算量
- 组合可以同时获得两种优势
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm

DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/outputs/output_quant_sample_combo")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def analyze_layer(layer_idx: int, block_size: int = 64, delta: float = 5.0, device: str = 'cuda') -> Dict:
    """分析单层的量化+采样组合方法"""
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

    # 采样索引
    sample1_idx = block_size // 2
    sample4_indices = [0, block_size//4, block_size//2, 3*block_size//4]

    # 初始化分数
    fp_scores = torch.zeros(HQ, num_blocks, device=device)

    # 单独方法
    fp_sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    fp_sample4_scores = torch.zeros(HQ, num_blocks, device=device)
    bit2_asym_scores = torch.zeros(HQ, num_blocks, device=device)
    bit1_asym_scores = torch.zeros(HQ, num_blocks, device=device)

    # 组合方法: 量化 + 采样
    bit2_asym_sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    bit2_asym_sample4_scores = torch.zeros(HQ, num_blocks, device=device)
    bit1_asym_sample1_scores = torch.zeros(HQ, num_blocks, device=device)
    bit1_asym_sample4_scores = torch.zeros(HQ, num_blocks, device=device)

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
        k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

        # ========== FP Ground Truth ==========
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

        # ========== FP + Sample-1 ==========
        k_s1 = k_block[:, sample1_idx, :]  # [num_blocks, K_dim]
        s1 = torch.einsum('gk,nk->gn', q_heads, k_s1) * scale
        fp_sample1_scores[kv_h * G : (kv_h + 1) * G] = s1

        # ========== FP + Sample-4 ==========
        k_s4 = k_block[:, sample4_indices, :]  # [num_blocks, 4, K_dim]
        s4 = torch.einsum('gk,nsk->gns', q_heads, k_s4) * scale
        fp_sample4_scores[kv_h * G : (kv_h + 1) * G] = s4.max(dim=-1).values

        # ========== 2-bit-asym 全量 ==========
        k_mean = k_block.mean(dim=1, keepdim=True)
        k_std = k_block.std(dim=1, keepdim=True).clamp(min=1e-6)
        k_scale_2bit = k_std * 2
        k_centered = k_block - k_mean
        k_q_2bit = torch.round(k_centered / k_scale_2bit + 1.5).clamp(0, 3)
        # 反量化
        k_dq_2bit = (k_q_2bit - 1.5) * k_scale_2bit + k_mean

        scores_2bit = torch.einsum('gk,nbk->gnb', q_heads, k_dq_2bit) * scale
        bit2_asym_scores[kv_h * G : (kv_h + 1) * G] = scores_2bit.max(dim=-1).values

        # ========== 1-bit-asym 全量 ==========
        k_1bit = k_mean + torch.sign(k_block - k_mean) * k_std
        scores_1bit = torch.einsum('gk,nbk->gnb', q_heads, k_1bit) * scale
        bit1_asym_scores[kv_h * G : (kv_h + 1) * G] = scores_1bit.max(dim=-1).values

        # ========== 2-bit-asym + Sample-1 ==========
        k_dq_2bit_s1 = k_dq_2bit[:, sample1_idx, :]
        s_2bit_s1 = torch.einsum('gk,nk->gn', q_heads, k_dq_2bit_s1) * scale
        bit2_asym_sample1_scores[kv_h * G : (kv_h + 1) * G] = s_2bit_s1

        # ========== 2-bit-asym + Sample-4 ==========
        k_dq_2bit_s4 = k_dq_2bit[:, sample4_indices, :]
        s_2bit_s4 = torch.einsum('gk,nsk->gns', q_heads, k_dq_2bit_s4) * scale
        bit2_asym_sample4_scores[kv_h * G : (kv_h + 1) * G] = s_2bit_s4.max(dim=-1).values

        # ========== 1-bit-asym + Sample-1 ==========
        k_1bit_s1 = k_1bit[:, sample1_idx, :]
        s_1bit_s1 = torch.einsum('gk,nk->gn', q_heads, k_1bit_s1) * scale
        bit1_asym_sample1_scores[kv_h * G : (kv_h + 1) * G] = s_1bit_s1

        # ========== 1-bit-asym + Sample-4 ==========
        k_1bit_s4 = k_1bit[:, sample4_indices, :]
        s_1bit_s4 = torch.einsum('gk,nsk->gns', q_heads, k_1bit_s4) * scale
        bit1_asym_sample4_scores[kv_h * G : (kv_h + 1) * G] = s_1bit_s4.max(dim=-1).values

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
        # 单独方法
        'FP-Sample1': fp_sample1_scores,
        'FP-Sample4': fp_sample4_scores,
        '2bit-asym': bit2_asym_scores,
        '1bit-asym': bit1_asym_scores,
        # 组合方法
        '2bit-Sample1': bit2_asym_sample1_scores,
        '2bit-Sample4': bit2_asym_sample4_scores,
        '1bit-Sample1': bit1_asym_sample1_scores,
        '1bit-Sample4': bit1_asym_sample4_scores,
    }

    for name, scores in methods.items():
        prune = scores < threshold
        agreement = (prune == fp_prune).float().mean().item()
        false_pos = ((prune) & (~fp_prune)).float().mean().item()
        false_neg = ((~prune) & (fp_prune)).float().mean().item()

        results[f'{name}_agreement'] = agreement
        results[f'{name}_false_pos'] = false_pos
        results[f'{name}_false_neg'] = false_neg

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

    methods = [
        'FP-Sample1', 'FP-Sample4', '2bit-asym', '1bit-asym',
        '2bit-Sample1', '2bit-Sample4', '1bit-Sample1', '1bit-Sample4'
    ]

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
    print("分析量化+采样组合方法的效果...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_sizes = [64, 128, 256, 512]

    single_methods = ['FP-Sample1', 'FP-Sample4', '2bit-asym', '1bit-asym']
    combo_methods = ['2bit-Sample1', '2bit-Sample4', '1bit-Sample1', '1bit-Sample4']
    all_methods = single_methods + combo_methods

    all_bs_results = {}

    for bs in block_sizes:
        avg_results, layer14_result = analyze_block_size(bs, device=device)
        all_bs_results[bs] = {
            'avg': avg_results,
            'l14': layer14_result
        }

    # 生成报告
    print("\n" + "=" * 100)
    print("量化+采样组合方法对比")
    print("=" * 100)

    # 按 Block Size 输出
    for bs in block_sizes:
        avg = all_bs_results[bs]['avg']
        l14 = all_bs_results[bs]['l14']
        print(f"\n### Block Size = {bs}")
        print(f"{'方法':<20} {'平均一致率':>12} {'L14一致率':>12} {'平均误剪率':>12}")
        print("-" * 60)

        print("--- 单独方法 ---")
        for method in single_methods:
            avg_agr = avg[f'{method}_agreement'] * 100
            l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
            avg_fp = avg[f'{method}_false_pos'] * 100
            print(f"{method:<20} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}%")

        print("--- 组合方法 ---")
        for method in combo_methods:
            avg_agr = avg[f'{method}_agreement'] * 100
            l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
            avg_fp = avg[f'{method}_false_pos'] * 100
            print(f"{method:<20} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}%")

    # 输出趋势表
    print("\n" + "=" * 100)
    print("各方法随 Block Size 变化趋势 (平均一致率)")
    print("=" * 100)
    print(f"\n{'方法':<20} {'BS=64':>10} {'BS=128':>10} {'BS=256':>10} {'BS=512':>10} {'衰减':>10}")
    print("-" * 70)
    for method in all_methods:
        values = [all_bs_results[bs]['avg'][f'{method}_agreement'] * 100 for bs in block_sizes]
        decay = values[0] - values[-1]
        print(f"{method:<20} {values[0]:>9.2f}% {values[1]:>9.2f}% {values[2]:>9.2f}% {values[3]:>9.2f}% {decay:>9.2f}%")

    # 保存报告
    report_path = OUTPUT_DIR / "quant_sample_combo_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("量化+采样组合方法对比分析\n")
        f.write("=" * 100 + "\n\n")

        f.write("方法说明:\n")
        f.write("- FP-Sample1/4: 仅采样，不量化 (减少计算量)\n")
        f.write("- 2bit/1bit-asym: 仅量化，不采样 (减少带宽)\n")
        f.write("- 2bit/1bit-Sample1/4: 量化+采样 (同时减少带宽和计算量)\n\n")

        f.write("存储/计算量对比 (相对于FP全量):\n")
        f.write("| 方法 | 存储 | 计算量 |\n")
        f.write("|------|------|--------|\n")
        f.write("| FP全量 | 16 bit | 100% |\n")
        f.write("| FP-Sample1 | 16 bit | 1/BS |\n")
        f.write("| FP-Sample4 | 16 bit | 4/BS |\n")
        f.write("| 2bit-asym | 2 bit | 100% |\n")
        f.write("| 1bit-asym | 1 bit | 100% |\n")
        f.write("| 2bit-Sample1 | 2 bit | 1/BS |\n")
        f.write("| 2bit-Sample4 | 2 bit | 4/BS |\n")
        f.write("| 1bit-Sample1 | 1 bit | 1/BS |\n")
        f.write("| 1bit-Sample4 | 1 bit | 4/BS |\n\n")

        for bs in block_sizes:
            avg = all_bs_results[bs]['avg']
            l14 = all_bs_results[bs]['l14']
            f.write(f"\n### Block Size = {bs}\n")
            f.write(f"| 方法 | 平均一致率 | L14一致率 | 平均误剪率 |\n")
            f.write(f"|------|-----------|----------|----------|\n")
            for method in all_methods:
                avg_agr = avg[f'{method}_agreement'] * 100
                l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
                avg_fp = avg[f'{method}_false_pos'] * 100
                f.write(f"| {method} | {avg_agr:.2f}% | {l14_agr:.2f}% | {avg_fp:.2f}% |\n")

        f.write("\n\n### 各方法随 Block Size 变化趋势 (平均一致率)\n")
        f.write(f"| 方法 | BS=64 | BS=128 | BS=256 | BS=512 | 衰减幅度 |\n")
        f.write(f"|------|-------|--------|--------|--------|----------|\n")
        for method in all_methods:
            values = [all_bs_results[bs]['avg'][f'{method}_agreement'] * 100 for bs in block_sizes]
            decay = values[0] - values[-1]
            f.write(f"| {method} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {values[3]:.2f}% | -{decay:.2f}% |\n")

    print(f"\n报告已保存到: {report_path}")

    return all_bs_results


if __name__ == "__main__":
    main()

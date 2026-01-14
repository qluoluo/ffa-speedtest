#!/usr/bin/env python3
"""
分析对称/非对称量化的不同组合:
- INT2-sym: 2-bit 对称量化 (当前实现)
- INT2-asym: 2-bit 非对称量化 (使用 mean 作为 zero-point)
- 1-bit-sym: 1-bit 对称量化 (以0为中心)
- 1-bit-asym: 1-bit 非对称量化 (当前实现，以 mean 为中心)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm

DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/outputs/output_quantization_variants")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def analyze_layer(layer_idx: int, block_size: int = 64, delta: float = 5.0, device: str = 'cuda') -> Dict:
    """分析单层的各量化方法一致率"""
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
    int2_sym_scores = torch.zeros(HQ, num_blocks, device=device)
    int2_asym_scores = torch.zeros(HQ, num_blocks, device=device)
    int1_sym_scores = torch.zeros(HQ, num_blocks, device=device)
    int1_asym_scores = torch.zeros(HQ, num_blocks, device=device)

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
        k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

        # 1. FP (ground truth)
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

        # 2. INT2-sym: 2-bit 对称量化 (以0为中心, scale = max_abs / 1.5)
        # 量化范围: [-1.5, -0.5, 0.5, 1.5] * scale
        k_max_abs = k_block.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
        k_scale_sym = k_max_abs / 1.5
        k_q_sym = torch.round(k_block / k_scale_sym + 1.5).clamp(0, 3)

        for g in range(G):
            q_g = q_heads[g]
            q_scaled = q_g.unsqueeze(0).unsqueeze(0) * k_scale_sym
            score_raw = (k_q_sym * q_scaled).sum(dim=-1)
            zp_offset = 1.5 * q_scaled.sum(dim=-1)
            int2_s = (score_raw - zp_offset) * scale
            int2_sym_scores[kv_h * G + g] = int2_s.max(dim=-1).values

        # 3. INT2-asym: 2-bit 非对称量化 (以 mean 为中心, scale = std * 3 / 1.5)
        # 量化范围: mean + [-1.5, -0.5, 0.5, 1.5] * scale
        k_mean = k_block.mean(dim=1, keepdim=True)  # [num_blocks, 1, K_dim]
        k_std = k_block.std(dim=1, keepdim=True).clamp(min=1e-6)
        k_scale_asym = k_std * 2  # 覆盖约 ±3 std 范围
        k_centered = k_block - k_mean
        k_q_asym = torch.round(k_centered / k_scale_asym + 1.5).clamp(0, 3)

        for g in range(G):
            q_g = q_heads[g]
            # 反量化: k_dq = (k_q - 1.5) * scale + mean
            # q · k_dq = q · ((k_q - 1.5) * scale + mean)
            #          = (k_q - 1.5) * (q · scale) + q · mean
            q_scaled = q_g.unsqueeze(0).unsqueeze(0) * k_scale_asym  # [num_blocks, 1, K_dim]
            score_raw = (k_q_asym * q_scaled).sum(dim=-1)  # [num_blocks, block_size]
            zp_offset = 1.5 * q_scaled.sum(dim=-1)  # [num_blocks, 1]
            mean_offset = (q_g.unsqueeze(0) * k_mean.squeeze(1)).sum(dim=-1, keepdim=True)  # [num_blocks, 1]
            int2_asym_s = (score_raw - zp_offset + mean_offset) * scale  # [num_blocks, block_size]
            int2_asym_scores[kv_h * G + g] = int2_asym_s.max(dim=-1).values

        # 4. 1-bit-sym: 1-bit 对称量化 (以0为中心)
        # k_1bit = sign(k) * std
        k_std_global = k_block.std(dim=1, keepdim=True)  # [num_blocks, 1, K_dim]
        k_1bit_sym = torch.sign(k_block) * k_std_global
        int1_sym_s = torch.einsum('gk,nbk->gnb', q_heads, k_1bit_sym) * scale
        int1_sym_scores[kv_h * G : (kv_h + 1) * G] = int1_sym_s.max(dim=-1).values

        # 5. 1-bit-asym: 1-bit 非对称量化 (以 mean 为中心)
        # k_1bit = mean + sign(k - mean) * std
        k_1bit_asym = k_mean + torch.sign(k_block - k_mean) * k_std
        int1_asym_s = torch.einsum('gk,nbk->gnb', q_heads, k_1bit_asym) * scale
        int1_asym_scores[kv_h * G : (kv_h + 1) * G] = int1_asym_s.max(dim=-1).values

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
        'INT2-sym': int2_sym_scores,
        'INT2-asym': int2_asym_scores,
        '1-bit-sym': int1_sym_scores,
        '1-bit-asym': int1_asym_scores,
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

    methods = ['INT2-sym', 'INT2-asym', '1-bit-sym', '1-bit-asym']

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
    print("分析对称/非对称量化的一致率...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_sizes = [64, 128, 256, 512]
    methods = ['INT2-sym', 'INT2-asym', '1-bit-sym', '1-bit-asym']

    all_bs_results = {}

    for bs in block_sizes:
        avg_results, layer14_result = analyze_block_size(bs, device=device)
        all_bs_results[bs] = {
            'avg': avg_results,
            'l14': layer14_result
        }

    # 生成报告
    print("\n" + "=" * 100)
    print("对称/非对称量化对比")
    print("=" * 100)

    # 按 Block Size 输出
    for bs in block_sizes:
        avg = all_bs_results[bs]['avg']
        l14 = all_bs_results[bs]['l14']
        print(f"\n### Block Size = {bs}")
        print(f"{'方法':<15} {'平均一致率':>12} {'L14一致率':>12} {'平均误剪率':>12} {'L14误剪率':>12}")
        print("-" * 65)
        for method in methods:
            avg_agr = avg[f'{method}_agreement'] * 100
            l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
            avg_fp = avg[f'{method}_false_pos'] * 100
            l14_fp = l14[f'{method}_false_pos'] * 100 if l14 else 0
            print(f"{method:<15} {avg_agr:>11.2f}% {l14_agr:>11.2f}% {avg_fp:>11.2f}% {l14_fp:>11.2f}%")

    # 按方法输出趋势
    print("\n" + "=" * 100)
    print("各方法随 Block Size 变化趋势 (平均一致率)")
    print("=" * 100)
    print(f"\n{'方法':<15} {'BS=64':>10} {'BS=128':>10} {'BS=256':>10} {'BS=512':>10} {'衰减':>10}")
    print("-" * 65)
    for method in methods:
        values = [all_bs_results[bs]['avg'][f'{method}_agreement'] * 100 for bs in block_sizes]
        decay = values[0] - values[-1]
        print(f"{method:<15} {values[0]:>9.2f}% {values[1]:>9.2f}% {values[2]:>9.2f}% {values[3]:>9.2f}% {decay:>9.2f}%")

    # 保存报告
    report_path = OUTPUT_DIR / "quantization_variants_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("对称/非对称量化对比分析\n")
        f.write("=" * 100 + "\n\n")

        f.write("方法说明:\n")
        f.write("- INT2-sym: 2-bit 对称量化 (zero-point=0, scale=max_abs/1.5)\n")
        f.write("- INT2-asym: 2-bit 非对称量化 (zero-point=mean, scale=std*2)\n")
        f.write("- 1-bit-sym: 1-bit 对称量化 (k' = sign(k) * std)\n")
        f.write("- 1-bit-asym: 1-bit 非对称量化 (k' = mean + sign(k-mean) * std)\n\n")

        for bs in block_sizes:
            avg = all_bs_results[bs]['avg']
            l14 = all_bs_results[bs]['l14']
            f.write(f"\n### Block Size = {bs}\n")
            f.write(f"| 方法 | 平均一致率 | L14一致率 | 平均误剪率 | L14误剪率 |\n")
            f.write(f"|------|-----------|----------|-----------|----------|\n")
            for method in methods:
                avg_agr = avg[f'{method}_agreement'] * 100
                l14_agr = l14[f'{method}_agreement'] * 100 if l14 else 0
                avg_fp = avg[f'{method}_false_pos'] * 100
                l14_fp = l14[f'{method}_false_pos'] * 100 if l14 else 0
                f.write(f"| {method} | {avg_agr:.2f}% | {l14_agr:.2f}% | {avg_fp:.2f}% | {l14_fp:.2f}% |\n")

        f.write("\n\n### 各方法随 Block Size 变化趋势 (平均一致率)\n")
        f.write(f"| 方法 | BS=64 | BS=128 | BS=256 | BS=512 | 衰减幅度 |\n")
        f.write(f"|------|-------|--------|--------|--------|----------|\n")
        for method in methods:
            values = [all_bs_results[bs]['avg'][f'{method}_agreement'] * 100 for bs in block_sizes]
            decay = values[0] - values[-1]
            f.write(f"| {method} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {values[3]:.2f}% | -{decay:.2f}% |\n")

    print(f"\n报告已保存到: {report_path}")

    return all_bs_results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
采样变体分析 - 优化版本
1. 不同采样数量: 1, 2, 4
2. 不同采样位置: 均匀分布, 前N个, 后N个
3. 采样+INT2组合
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import os

# 数据路径
DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/output_sampling_variants")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_layer_data(layer_idx: int, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """加载指定层的Q和K数据"""
    layer_dir = DATA_DIR / "layer_data" / f"layer_{layer_idx}"
    q_path = layer_dir / "q_rope.pt"
    k_path = layer_dir / "k_rope.pt"

    Q = torch.load(q_path, map_location='cpu').float().to(device)
    K = torch.load(k_path, map_location='cpu').float().to(device)

    return Q, K


def get_sample_indices(block_size: int, n_samples: int, strategy: str) -> List[int]:
    """获取采样索引"""
    if strategy == 'uniform':
        if n_samples == 1:
            return [block_size // 2]
        else:
            step = block_size / (n_samples + 1)
            return [int(step * (i + 1)) for i in range(n_samples)]
    elif strategy == 'first':
        return list(range(n_samples))
    elif strategy == 'last':
        return list(range(block_size - n_samples, block_size))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def analyze_layer_vectorized(layer_idx: int, block_size: int, delta: float = 5.0,
                             device: str = 'cuda') -> Dict:
    """向量化分析 - 大幅加速"""
    Q, K = load_layer_data(layer_idx, device)

    # 取最后一个query
    q = Q[0, :, -1, :]  # [HQ, K_dim]
    K = K[0]  # [HKV, T, K_dim]

    HQ, K_dim = q.shape
    HKV, T, _ = K.shape
    G = HQ // HKV
    num_blocks = T // block_size

    # 重塑K为blocks
    K_blocks = K[:, :num_blocks*block_size, :].reshape(HKV, num_blocks, block_size, K_dim)

    scale = 1.0 / np.sqrt(K_dim)

    # 扩展q以匹配KV头 [HQ, K_dim] -> 需要按GQA映射
    # 计算所有FP分数 (向量化)
    # 对每个KV头，计算其对应的G个Q头的分数
    fp_scores = torch.zeros(HQ, num_blocks, device=device)

    for kv_h in range(HKV):
        # 这个KV头对应的Q头
        q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
        k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

        # 计算分数 [G, num_blocks, block_size]
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale

        # 取每个block的最大值
        max_scores = scores.max(dim=-1).values  # [G, num_blocks]
        fp_scores[kv_h * G : (kv_h + 1) * G] = max_scores

    # 计算阈值
    first_block_scores = fp_scores[:, 0]
    last_block_scores = fp_scores[:, -1]
    threshold = torch.maximum(first_block_scores, last_block_scores) - delta
    threshold = threshold.unsqueeze(1)  # [HQ, 1]

    fp_prune = fp_scores < threshold
    fp_prune_rate = fp_prune.float().mean().item()

    results = {
        'fp_prune_rate': fp_prune_rate,
        'methods': {}
    }

    # 测试配置
    sample_configs = [
        (1, 'uniform', False),
        (2, 'uniform', False),
        (4, 'uniform', False),
        (4, 'first', False),
        (4, 'last', False),
        (1, 'uniform', True),
        (2, 'uniform', True),
        (4, 'uniform', True),
        (4, 'first', True),
        (4, 'last', True),
    ]

    for n_samples, strategy, use_int2 in sample_configs:
        indices = get_sample_indices(block_size, n_samples, strategy)

        if use_int2:
            method_name = f"Sample-{n_samples}-{strategy}-INT2"
        else:
            method_name = f"Sample-{n_samples}-{strategy}"

        method_scores = torch.zeros(HQ, num_blocks, device=device)

        for kv_h in range(HKV):
            q_heads = q[kv_h * G : (kv_h + 1) * G]  # [G, K_dim]
            k_block = K_blocks[kv_h]  # [num_blocks, block_size, K_dim]

            # 采样
            k_sampled = k_block[:, indices, :]  # [num_blocks, n_samples, K_dim]

            if use_int2:
                # INT2量化
                k_max_abs = k_sampled.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)  # [num_blocks, 1, K_dim]
                k_scale = k_max_abs / 1.5
                k_q = torch.round(k_sampled / k_scale + 1.5).clamp(0, 3)

                # 对每个Q头计算
                for g in range(G):
                    q_g = q_heads[g]  # [K_dim]
                    q_scaled = q_g.unsqueeze(0).unsqueeze(0) * k_scale  # [num_blocks, 1, K_dim]
                    score_raw = (k_q * q_scaled).sum(dim=-1)  # [num_blocks, n_samples]
                    zp_offset = 1.5 * q_scaled.sum(dim=-1)  # [num_blocks, 1]
                    scores = (score_raw - zp_offset) * scale
                    method_scores[kv_h * G + g] = scores.max(dim=-1).values
            else:
                # FP采样
                scores = torch.einsum('gk,nsk->gns', q_heads, k_sampled) * scale  # [G, num_blocks, n_samples]
                max_scores = scores.max(dim=-1).values  # [G, num_blocks]
                method_scores[kv_h * G : (kv_h + 1) * G] = max_scores

        # 计算统计
        prune = method_scores < threshold
        prune_rate = prune.float().mean().item()
        agreement = (prune == fp_prune).float().mean().item()
        false_positive = ((prune) & (~fp_prune)).float().mean().item()
        false_negative = ((~prune) & (fp_prune)).float().mean().item()

        results['methods'][method_name] = {
            'prune_rate': prune_rate,
            'agreement': agreement,
            'false_positive': false_positive,
            'false_negative': false_negative,
        }

    return results


def main():
    print("=" * 80)
    print("采样变体分析")
    print("=" * 80)

    block_sizes = [8, 16, 32, 64]
    layer_idx = 14

    all_results = {}

    for bs in block_sizes:
        print(f"\n分析 Block Size = {bs}...")
        results = analyze_layer_vectorized(layer_idx, bs)
        all_results[bs] = results
        print(f"  FP剪枝率: {results['fp_prune_rate']*100:.1f}%")

    # 生成报告
    report_path = OUTPUT_DIR / "sampling_variants_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("采样变体分析报告\n")
        f.write("=" * 80 + "\n\n")

        for bs in block_sizes:
            f.write("=" * 60 + "\n")
            f.write(f"Block Size = {bs}\n")
            f.write("=" * 60 + "\n\n")

            results = all_results[bs]
            f.write(f"FP Ground Truth 剪枝率: {results['fp_prune_rate']*100:.1f}%\n\n")

            f.write(f"{'方法':<30} {'剪枝率':>8} {'一致率':>8} {'误剪率':>8} {'漏剪率':>8}\n")
            f.write("-" * 70 + "\n")

            for method_name, stats in results['methods'].items():
                f.write(f"{method_name:<30} {stats['prune_rate']*100:>7.1f}% "
                       f"{stats['agreement']*100:>7.1f}% "
                       f"{stats['false_positive']*100:>7.2f}% "
                       f"{stats['false_negative']*100:>7.1f}%\n")
            f.write("\n")

        # 添加关键发现
        f.write("\n" + "=" * 80 + "\n")
        f.write("关键发现\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 采样位置对比 (Block Size = 64):\n")
        bs = 64
        results = all_results[bs]
        f.write(f"   - 均匀采样4点: 一致率 {results['methods']['Sample-4-uniform']['agreement']*100:.1f}%\n")
        f.write(f"   - 前4个位置:   一致率 {results['methods']['Sample-4-first']['agreement']*100:.1f}%\n")
        f.write(f"   - 后4个位置:   一致率 {results['methods']['Sample-4-last']['agreement']*100:.1f}%\n\n")

        f.write("2. 采样+INT2 vs 纯采样:\n")
        for bs in block_sizes:
            results = all_results[bs]
            fp_4 = results['methods']['Sample-4-uniform']['agreement'] * 100
            int2_4 = results['methods']['Sample-4-uniform-INT2']['agreement'] * 100
            f.write(f"   BS={bs}: FP采样={fp_4:.1f}%, INT2采样={int2_4:.1f}%, 差异={fp_4-int2_4:.2f}%\n")

        f.write("\n3. 采样数量对比 (Block Size = 64):\n")
        for n in [1, 2, 4]:
            method = f"Sample-{n}-uniform"
            agreement = results['methods'][method]['agreement'] * 100
            f.write(f"   - {n}点采样: 一致率 {agreement:.1f}%\n")

    print(f"\n报告已保存到: {report_path}")

    # 生成对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fp_methods = ['Sample-1-uniform', 'Sample-2-uniform', 'Sample-4-uniform',
                  'Sample-4-first', 'Sample-4-last']
    int2_methods = ['Sample-1-uniform-INT2', 'Sample-2-uniform-INT2', 'Sample-4-uniform-INT2',
                    'Sample-4-first-INT2', 'Sample-4-last-INT2']

    for idx, bs in enumerate(block_sizes):
        ax = axes[idx // 2, idx % 2]
        results = all_results[bs]

        x = np.arange(len(fp_methods))
        width = 0.35

        fp_agreements = [results['methods'][m]['agreement'] * 100 for m in fp_methods]
        int2_agreements = [results['methods'][m]['agreement'] * 100 for m in int2_methods]

        bars1 = ax.bar(x - width/2, fp_agreements, width, label='FP采样', color='steelblue')
        bars2 = ax.bar(x + width/2, int2_agreements, width, label='采样+INT2', color='coral')

        ax.set_ylabel('一致率 (%)')
        ax.set_title(f'Block Size = {bs}')
        ax.set_xticks(x)
        ax.set_xticklabels(['1-均匀', '2-均匀', '4-均匀', '4-前', '4-后'], rotation=45, ha='right')
        ax.set_ylim(85, 100)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.suptitle('采样变体对比: FP采样 vs 采样+INT2', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sampling_variants_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比图已保存到: {OUTPUT_DIR / 'sampling_variants_comparison.png'}")

    # 打印关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    bs = 64
    results = all_results[bs]

    print(f"\nBlock Size = {bs} 时的一致率对比:")
    print(f"{'方法':<30} {'一致率':>10} {'误剪率':>10}")
    print("-" * 52)

    sorted_methods = sorted(results['methods'].items(),
                           key=lambda x: x[1]['agreement'], reverse=True)
    for method_name, stats in sorted_methods:
        print(f"{method_name:<30} {stats['agreement']*100:>9.2f}% {stats['false_positive']*100:>9.2f}%")


if __name__ == "__main__":
    main()

"""
探索其他粗粒度筛选方法

除了MinMax和Norm上界，这里实现并测试更多的粗粒度筛选方法：

1. 质心法 (Centroid): 用block的K均值近似整个block
2. 采样法 (Sampling): 从block中采样几个K向量
3. 符号法 (Sign): 只用K的符号信息
4. 紧凑上界法 (Tight Bound): 结合多种约束得到更紧的上界
5. 低秩近似法 (Low Rank): 用SVD主成分近似
6. 1-bit量化法: 更极端的二值化量化
"""

import os
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm import tqdm

DATA_ROOT = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/experiments/consistency_analysis/outputs/new_methods")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def analyze_new_methods(
    layer_idx: int,
    block_size: int = 16,
    delta: float = 5.0,
    device: str = 'cuda',
) -> Dict:
    """
    测试多种粗粒度筛选方法
    """
    layer_path = DATA_ROOT / f"layer_{layer_idx}"

    q = torch.load(layer_path / "q_rope.pt", weights_only=True, map_location='cpu')
    k = torch.load(layer_path / "k_rope.pt", weights_only=True, map_location='cpu')

    q = q.float().to(device)
    k = k.float().to(device)

    B, HQ, T, K_dim = q.shape
    _, HKV, _, _ = k.shape
    G = HQ // HKV

    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)

    q_single = q[:, :, -1, :]  # [B, HQ, K]

    # Padding
    pad_len = num_blocks * block_size - T
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len), value=0)

    k_blocks = k.view(B, HKV, num_blocks, block_size, K_dim)

    # 初始化各方法的分数
    scores = {
        'fp': torch.zeros(HQ, num_blocks, device=device),
        'centroid': torch.zeros(HQ, num_blocks, device=device),
        'sample_1': torch.zeros(HQ, num_blocks, device=device),  # 采样1个
        'sample_4': torch.zeros(HQ, num_blocks, device=device),  # 采样4个
        'minmax': torch.zeros(HQ, num_blocks, device=device),
        'sign': torch.zeros(HQ, num_blocks, device=device),
        'tight_bound': torch.zeros(HQ, num_blocks, device=device),
        'mean_std': torch.zeros(HQ, num_blocks, device=device),  # 均值+标准差上界
        'int1': torch.zeros(HQ, num_blocks, device=device),  # 1-bit量化
    }

    for hkv in range(HKV):
        k_hkv = k_blocks[:, hkv]  # [B, num_blocks, block_size, K]

        # === 预计算block统计量 ===
        k_mean = k_hkv.mean(dim=2)  # [B, num_blocks, K] - 质心
        k_std = k_hkv.std(dim=2)    # [B, num_blocks, K] - 标准差
        k_min = k_hkv.amin(dim=2)   # [B, num_blocks, K]
        k_max = k_hkv.amax(dim=2)   # [B, num_blocks, K]
        k_norms = k_hkv.norm(dim=-1)  # [B, num_blocks, block_size]
        k_norm_max = k_norms.amax(dim=-1)  # [B, num_blocks]
        k_norm_mean = k_norms.mean(dim=-1)

        # 1-bit量化: sign(k - mean)
        k_sign = torch.sign(k_hkv - k_mean.unsqueeze(2))  # [B, num_blocks, block_size, K]

        for g in range(G):
            hq = hkv * G + g
            q_vec = q_single[:, hq, :]  # [B, K]
            q_norm = q_vec.norm(dim=-1)
            q_abs = q_vec.abs()

            # ====== 1. FP (Ground Truth) ======
            scores_fp = torch.einsum('bk,bntk->bnt', q_vec, k_hkv) * scale
            fp_max = scores_fp.amax(dim=-1)
            scores['fp'][hq] = fp_max[0]

            # ====== 2. 质心法 (Centroid) ======
            # 用block均值近似: score ≈ q · k_mean
            centroid_score = torch.einsum('bk,bnk->bn', q_vec, k_mean) * scale
            scores['centroid'][hq] = centroid_score[0]

            # ====== 3. 采样法 ======
            # 采样1个(中间位置)
            mid_idx = block_size // 2
            k_sample_1 = k_hkv[:, :, mid_idx, :]  # [B, num_blocks, K]
            sample_1_score = torch.einsum('bk,bnk->bn', q_vec, k_sample_1) * scale
            scores['sample_1'][hq] = sample_1_score[0]

            # 采样4个(均匀分布)并取max
            sample_indices = [0, block_size//4, block_size//2, 3*block_size//4]
            sample_scores = []
            for idx in sample_indices:
                if idx < block_size:
                    k_s = k_hkv[:, :, idx, :]
                    s = torch.einsum('bk,bnk->bn', q_vec, k_s) * scale
                    sample_scores.append(s)
            sample_4_score = torch.stack(sample_scores, dim=-1).amax(dim=-1)
            scores['sample_4'][hq] = sample_4_score[0]

            # ====== 4. MinMax上界 ======
            k_opt = torch.where(q_vec[:, None, :] > 0, k_max, k_min)
            minmax_bound = (q_vec[:, None, :] * k_opt).sum(dim=-1) * scale
            scores['minmax'][hq] = minmax_bound[0]

            # ====== 5. 符号法 (Sign) ======
            # 用符号信息估计: score ≈ q · (k_mean + |q|·sign(q)·k_std)
            # 这是一个启发式上界
            q_sign = torch.sign(q_vec)
            k_adjusted = k_mean + q_sign[:, None, :] * k_std * 1.5  # 1.5倍标准差
            sign_score = torch.einsum('bk,bnk->bn', q_vec, k_adjusted) * scale
            scores['sign'][hq] = sign_score[0]

            # ====== 6. 紧凑上界法 (Tight Bound) ======
            # 结合MinMax和Norm: min(minmax_bound, norm_bound)
            norm_bound = q_norm[:, None] * k_norm_max * scale
            tight_bound = torch.minimum(minmax_bound, norm_bound)
            scores['tight_bound'][hq] = tight_bound[0]

            # ====== 7. 均值+标准差上界 ======
            # upper_bound = q·k_mean + |q|·k_std * sqrt(block_size)
            # 基于: max(q·k) ≤ q·mean(k) + ||q||·std(k)·sqrt(n) (统计上界)
            base_score = torch.einsum('bk,bnk->bn', q_vec, k_mean)
            std_contrib = (q_abs[:, None, :] * k_std).sum(dim=-1) * math.sqrt(block_size) * 0.5
            mean_std_bound = (base_score + std_contrib) * scale
            scores['mean_std'][hq] = mean_std_bound[0]

            # ====== 8. 1-bit量化 ======
            # k_1bit = k_mean + sign(k - k_mean) * k_std
            # 然后计算 q · k_1bit
            k_1bit = k_mean.unsqueeze(2) + k_sign * k_std.unsqueeze(2)  # [B, num_blocks, block_size, K]
            scores_1bit = torch.einsum('bk,bntk->bnt', q_vec, k_1bit) * scale
            int1_max = scores_1bit.amax(dim=-1)
            scores['int1'][hq] = int1_max[0]

    # 转numpy
    for key in scores:
        scores[key] = scores[key].cpu().numpy()

    # 计算阈值和剪枝mask
    fp_threshold = np.maximum(scores['fp'][:, 0], scores['fp'][:, -1]) - delta

    results = {
        'layer_idx': layer_idx,
        'block_size': block_size,
        'num_blocks': num_blocks,
        'scores': scores,
        'threshold': fp_threshold,
    }

    # 计算各方法的统计
    for method in scores:
        if method == 'fp':
            continue

        # 对于上界方法(minmax, tight_bound, mean_std, sign)，使用上界判断
        # 对于近似方法(centroid, sample, int1)，使用近似分数判断

        is_upper_bound = method in ['minmax', 'tight_bound', 'mean_std', 'sign']

        if is_upper_bound:
            # 上界方法: 如果上界 < 阈值，则剪掉
            method_prune = scores[method] < fp_threshold[:, None]
        else:
            # 近似方法: 需要用近似分数的阈值
            method_threshold = np.maximum(scores[method][:, 0], scores[method][:, -1]) - delta
            method_prune = scores[method] < method_threshold[:, None]

        fp_prune = scores['fp'] < fp_threshold[:, None]

        # 统计
        agreement = (method_prune == fp_prune).mean()
        false_pos = (method_prune & ~fp_prune).mean()  # 误剪
        false_neg = (~method_prune & fp_prune).mean()  # 漏剪

        results[f'{method}_prune_ratio'] = method_prune.mean()
        results[f'{method}_agreement'] = agreement
        results[f'{method}_false_pos'] = false_pos
        results[f'{method}_false_neg'] = false_neg

        if not is_upper_bound:
            # 计算分数误差
            error = np.abs(scores[method] - scores['fp']).mean()
            results[f'{method}_error'] = error

    results['fp_prune_ratio'] = (scores['fp'] < fp_threshold[:, None]).mean()

    return results


def run_analysis():
    """运行所有层的分析"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    available_layers = sorted([int(p.name.split('_')[1]) for p in DATA_ROOT.iterdir() if p.is_dir()])

    # 测试不同block size
    block_sizes = [8, 16, 32, 64]

    all_results = {}

    for bs in block_sizes:
        print(f"\n{'='*60}")
        print(f"Block Size: {bs}")
        print(f"{'='*60}")

        bs_results = []
        for layer_idx in tqdm(available_layers, desc=f"BS={bs}"):
            try:
                result = analyze_new_methods(layer_idx, bs, delta=5.0, device=device)
                bs_results.append(result)
            except Exception as e:
                print(f"Layer {layer_idx} error: {e}")

        all_results[bs] = bs_results

    return all_results


def visualize_results(all_results: Dict):
    """可视化各方法对比"""
    block_sizes = sorted(all_results.keys())

    # 方法列表
    methods = ['centroid', 'sample_1', 'sample_4', 'minmax', 'sign',
               'tight_bound', 'mean_std', 'int1']

    method_names = {
        'centroid': 'Centroid (Mean)',
        'sample_1': 'Sample (1 point)',
        'sample_4': 'Sample (4 points)',
        'minmax': 'MinMax Bound',
        'sign': 'Sign+Std Bound',
        'tight_bound': 'Tight Bound',
        'mean_std': 'Mean+Std Bound',
        'int1': '1-bit Quantization',
    }

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for bs in block_sizes:
        bs_results = all_results[bs]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Block Size = {bs}: New Filtering Methods Comparison', fontsize=14)

        layers = [r['layer_idx'] for r in bs_results]

        # 1. 剪枝率对比
        ax1 = axes[0, 0]
        fp_prune = [r['fp_prune_ratio'] * 100 for r in bs_results]
        ax1.plot(layers, fp_prune, 'k-', linewidth=2, label='FP (GT)', marker='o', markersize=4)

        for i, method in enumerate(methods):
            prune = [r.get(f'{method}_prune_ratio', 0) * 100 for r in bs_results]
            ax1.plot(layers, prune, color=colors[i], label=method_names[method],
                    alpha=0.7, marker='s', markersize=3)

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Prune Ratio (%)')
        ax1.set_title('Prune Ratio by Method')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(alpha=0.3)

        # 2. 与FP一致率
        ax2 = axes[0, 1]
        for i, method in enumerate(methods):
            agreement = [r.get(f'{method}_agreement', 0) * 100 for r in bs_results]
            ax2.plot(layers, agreement, color=colors[i], label=method_names[method],
                    alpha=0.7, marker='s', markersize=3)

        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Agreement with FP (%)')
        ax2.set_title('Agreement Rate with Ground Truth')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(alpha=0.3)

        # 3. 误剪率
        ax3 = axes[1, 0]
        for i, method in enumerate(methods):
            false_pos = [r.get(f'{method}_false_pos', 0) * 100 for r in bs_results]
            ax3.plot(layers, false_pos, color=colors[i], label=method_names[method],
                    alpha=0.7, marker='s', markersize=3)

        ax3.set_xlabel('Layer')
        ax3.set_ylabel('False Positive Rate (%)')
        ax3.set_title('False Positive Rate (Dangerous: Prune when should Keep)')
        ax3.legend(fontsize=7, ncol=2)
        ax3.grid(alpha=0.3)

        # 4. 汇总统计
        ax4 = axes[1, 1]

        summary_text = f"Block Size = {bs}\n"
        summary_text += f"Num Blocks: {bs_results[0]['num_blocks']}\n"
        summary_text += f"FP Avg Prune: {np.mean(fp_prune):.1f}%\n\n"
        summary_text += f"{'Method':<20} {'Prune%':>8} {'Agree%':>8} {'FP%':>8} {'FN%':>8}\n"
        summary_text += "-" * 55 + "\n"

        for method in methods:
            prune = np.mean([r.get(f'{method}_prune_ratio', 0) * 100 for r in bs_results])
            agree = np.mean([r.get(f'{method}_agreement', 0) * 100 for r in bs_results])
            fp = np.mean([r.get(f'{method}_false_pos', 0) * 100 for r in bs_results])
            fn = np.mean([r.get(f'{method}_false_neg', 0) * 100 for r in bs_results])
            summary_text += f"{method_names[method]:<20} {prune:>8.1f} {agree:>8.1f} {fp:>8.2f} {fn:>8.1f}\n"

        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'methods_comparison_bs{bs}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {OUTPUT_DIR / f'methods_comparison_bs{bs}.png'}")


def create_summary_report(all_results: Dict):
    """创建汇总报告"""
    block_sizes = sorted(all_results.keys())

    methods = ['centroid', 'sample_1', 'sample_4', 'minmax', 'sign',
               'tight_bound', 'mean_std', 'int1']

    method_names = {
        'centroid': 'Centroid (质心法)',
        'sample_1': 'Sample-1 (单点采样)',
        'sample_4': 'Sample-4 (4点采样)',
        'minmax': 'MinMax (min/max上界)',
        'sign': 'Sign+Std (符号+标准差)',
        'tight_bound': 'Tight (紧凑上界)',
        'mean_std': 'Mean+Std (均值+标准差)',
        'int1': '1-bit (二值量化)',
    }

    with open(OUTPUT_DIR / 'new_methods_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("新型粗粒度筛选方法对比分析\n")
        f.write("=" * 80 + "\n\n")

        for bs in block_sizes:
            bs_results = all_results[bs]

            f.write(f"\n{'='*60}\n")
            f.write(f"Block Size = {bs}\n")
            f.write(f"{'='*60}\n\n")

            fp_prune = np.mean([r['fp_prune_ratio'] * 100 for r in bs_results])
            f.write(f"FP Ground Truth 剪枝率: {fp_prune:.1f}%\n\n")

            f.write(f"{'方法':<25} {'剪枝率':>10} {'一致率':>10} {'误剪率':>10} {'漏剪率':>10}\n")
            f.write("-" * 70 + "\n")

            for method in methods:
                prune = np.mean([r.get(f'{method}_prune_ratio', 0) * 100 for r in bs_results])
                agree = np.mean([r.get(f'{method}_agreement', 0) * 100 for r in bs_results])
                fp = np.mean([r.get(f'{method}_false_pos', 0) * 100 for r in bs_results])
                fn = np.mean([r.get(f'{method}_false_neg', 0) * 100 for r in bs_results])

                name = method_names[method]
                f.write(f"{name:<25} {prune:>10.1f}% {agree:>10.1f}% {fp:>10.2f}% {fn:>10.1f}%\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("方法说明\n")
        f.write("=" * 80 + "\n\n")

        f.write("""
1. Centroid (质心法)
   - 原理: 用block的K向量均值近似整个block
   - 计算: score ≈ q · mean(K_block)
   - 特点: 计算简单，但只能给出估计值，不是上界

2. Sample (采样法)
   - 原理: 从block中采样几个K向量，取最大分数
   - Sample-1: 只取中间位置
   - Sample-4: 取4个均匀分布的位置
   - 特点: 计算量与采样数成正比

3. MinMax (min/max上界)
   - 原理: 对每个维度取最优的min或max值
   - 计算: upper = sum(q[d] * k_opt[d]), k_opt[d] = max if q[d]>0 else min
   - 特点: 保守上界，零误剪，但可能很松

4. Sign+Std (符号+标准差)
   - 原理: 用均值加符号调整的标准差估计上界
   - 计算: upper ≈ q · (k_mean + sign(q) * k_std * 1.5)
   - 特点: 启发式方法，比MinMax更紧

5. Tight Bound (紧凑上界)
   - 原理: 取MinMax和Norm上界的最小值
   - 计算: upper = min(minmax_bound, ||q|| * ||k||_max)
   - 特点: 结合两种上界的优点

6. Mean+Std (均值+标准差上界)
   - 原理: 基于统计学的上界估计
   - 计算: upper = q·k_mean + |q|·k_std * sqrt(n) * factor
   - 特点: 考虑了分布信息

7. 1-bit (二值量化)
   - 原理: 将K量化为 mean ± std
   - 计算: k_1bit = k_mean + sign(k - k_mean) * k_std
   - 特点: 极端压缩，保留符号信息
""")

    print(f"Saved: {OUTPUT_DIR / 'new_methods_report.txt'}")


def main():
    print("=" * 60)
    print("探索新型粗粒度筛选方法")
    print("=" * 60)

    all_results = run_analysis()

    print("\n生成可视化...")
    visualize_results(all_results)
    create_summary_report(all_results)

    # 保存原始结果
    torch.save(all_results, OUTPUT_DIR / 'all_results.pt')

    print("\n" + "=" * 60)
    print(f"分析完成! 结果保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

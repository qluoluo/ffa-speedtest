"""
Detailed analysis of coarse filtering bounds tightness.
This script analyzes why MinMax and Norm bounds are too loose for effective pruning.
"""

import os
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm import tqdm

# Data path configuration
DATA_ROOT = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data")
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/output")


def analyze_bound_tightness(layer_idx: int = 14, device: str = 'cuda'):
    """
    Analyze the tightness of different upper bounds for attention scores.
    """
    layer_path = DATA_ROOT / f"layer_{layer_idx}"

    # Load data
    q = torch.load(layer_path / "q_rope.pt", weights_only=True, map_location='cpu')
    k = torch.load(layer_path / "k_rope.pt", weights_only=True, map_location='cpu')

    q = q.float().to(device)
    k = k.float().to(device)

    B, HQ, T, K_dim = q.shape
    _, HKV, _, _ = k.shape
    G = HQ // HKV

    block_size = 128
    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)

    # Take the last query
    q_single = q[:, :, -1, :]  # [B, HQ, K]

    print(f"Layer {layer_idx}: T={T}, HQ={HQ}, HKV={HKV}, K={K_dim}")
    print(f"Num blocks: {num_blocks}, Block size: {block_size}")

    # Compute statistics for each block
    results = {
        'fp_scores': [],          # Actual max scores per block
        'minmax_bounds': [],      # MinMax upper bound
        'norm_bounds': [],        # Norm upper bound
        'k_norm_max': [],         # Max K norm in block
        'k_norm_mean': [],        # Mean K norm in block
        'q_norm': [],             # Q norm
        'gap_minmax': [],         # Gap between MinMax bound and FP
        'gap_norm': [],           # Gap between Norm bound and FP
    }

    for blk_idx in tqdm(range(num_blocks), desc="Analyzing blocks"):
        start = blk_idx * block_size
        end = min(start + block_size, T)

        k_blk = k[:, :, start:end, :]  # [B, HKV, blk_len, K]

        # Per-dimension min/max
        k_min = k_blk.amin(dim=2)  # [B, HKV, K]
        k_max = k_blk.amax(dim=2)

        # Norms
        k_norms = k_blk.norm(dim=-1)  # [B, HKV, blk_len]
        k_norm_max = k_norms.amax(dim=-1)  # [B, HKV]
        k_norm_mean = k_norms.mean(dim=-1)

        for hkv in range(HKV):
            for g in range(G):
                hq = hkv * G + g
                q_vec = q_single[:, hq, :]  # [B, K]

                # Actual FP max score
                scores = torch.einsum('bk,btk->bt', q_vec, k_blk[:, hkv]) * scale
                fp_max = scores.max(dim=-1).values  # [B]

                # MinMax upper bound
                k_opt = torch.where(q_vec > 0, k_max[:, hkv], k_min[:, hkv])
                minmax_bound = (q_vec * k_opt).sum(dim=-1) * scale

                # Norm upper bound
                q_norm = q_vec.norm(dim=-1)
                norm_bound = q_norm * k_norm_max[:, hkv] * scale

                results['fp_scores'].append(fp_max.item())
                results['minmax_bounds'].append(minmax_bound.item())
                results['norm_bounds'].append(norm_bound.item())
                results['k_norm_max'].append(k_norm_max[:, hkv].item())
                results['k_norm_mean'].append(k_norm_mean[:, hkv].item())
                results['q_norm'].append(q_norm.item())
                results['gap_minmax'].append((minmax_bound - fp_max).item())
                results['gap_norm'].append((norm_bound - fp_max).item())

    # Convert to numpy
    for key in results:
        results[key] = np.array(results[key])

    return results


def visualize_tightness(results: Dict, output_path: Path, layer_idx: int):
    """
    Visualize the bound tightness analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Score distributions
    ax1 = axes[0, 0]
    ax1.hist(results['fp_scores'], bins=50, alpha=0.5, label='FP Actual', color='blue', density=True)
    ax1.hist(results['minmax_bounds'], bins=50, alpha=0.5, label='MinMax Bound', color='orange', density=True)
    ax1.hist(results['norm_bounds'], bins=50, alpha=0.5, label='Norm Bound', color='red', density=True)
    ax1.set_xlabel('Attention Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution Comparison')
    ax1.legend()

    # 2. Gap analysis
    ax2 = axes[0, 1]
    ax2.hist(results['gap_minmax'], bins=50, alpha=0.7, label='MinMax Gap', color='orange', density=True)
    ax2.hist(results['gap_norm'], bins=50, alpha=0.7, label='Norm Gap', color='red', density=True)
    ax2.axvline(x=0, color='k', linestyle='--')
    ax2.set_xlabel('Upper Bound - FP Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Bound Gap Distribution')
    ax2.legend()

    # 3. Scatter: FP vs bounds
    ax3 = axes[0, 2]
    # Sample for visualization
    n_samples = min(5000, len(results['fp_scores']))
    idx = np.random.choice(len(results['fp_scores']), n_samples, replace=False)
    ax3.scatter(results['fp_scores'][idx], results['minmax_bounds'][idx], alpha=0.2, s=1, label='MinMax', color='orange')
    ax3.scatter(results['fp_scores'][idx], results['norm_bounds'][idx], alpha=0.2, s=1, label='Norm', color='red')
    ax3.plot([results['fp_scores'].min(), results['fp_scores'].max()],
             [results['fp_scores'].min(), results['fp_scores'].max()], 'k--', label='y=x')
    ax3.set_xlabel('FP Actual Score')
    ax3.set_ylabel('Upper Bound')
    ax3.set_title('FP Score vs Upper Bounds')
    ax3.legend()

    # 4. Gap vs FP score
    ax4 = axes[1, 0]
    ax4.scatter(results['fp_scores'][idx], results['gap_minmax'][idx], alpha=0.2, s=1, label='MinMax Gap', color='orange')
    ax4.scatter(results['fp_scores'][idx], results['gap_norm'][idx], alpha=0.2, s=1, label='Norm Gap', color='red')
    ax4.axhline(y=0, color='k', linestyle='--')
    ax4.set_xlabel('FP Actual Score')
    ax4.set_ylabel('Bound Gap')
    ax4.set_title('Gap vs FP Score')
    ax4.legend()

    # 5. K norm analysis
    ax5 = axes[1, 1]
    ax5.scatter(results['k_norm_max'][idx], results['gap_norm'][idx], alpha=0.2, s=1, color='purple')
    ax5.set_xlabel('K Norm Max (per block)')
    ax5.set_ylabel('Norm Bound Gap')
    ax5.set_title('K Norm vs Gap')

    # 6. Summary statistics
    ax6 = axes[1, 2]
    stats_text = f"""Bound Tightness Analysis (Layer {layer_idx})

    Sample size: {len(results['fp_scores'])}

    FP Scores:
      Mean: {results['fp_scores'].mean():.3f}
      Std: {results['fp_scores'].std():.3f}
      Min: {results['fp_scores'].min():.3f}
      Max: {results['fp_scores'].max():.3f}

    MinMax Bound Gap:
      Mean: {results['gap_minmax'].mean():.3f}
      Std: {results['gap_minmax'].std():.3f}
      Min: {results['gap_minmax'].min():.3f}
      Max: {results['gap_minmax'].max():.3f}

    Norm Bound Gap:
      Mean: {results['gap_norm'].mean():.3f}
      Std: {results['gap_norm'].std():.3f}
      Min: {results['gap_norm'].min():.3f}
      Max: {results['gap_norm'].max():.3f}

    Gap Ratio (Upper Bound / FP):
      MinMax: {(results['minmax_bounds'] / (results['fp_scores'] + 1e-8)).mean():.2f}x
      Norm: {(results['norm_bounds'] / (results['fp_scores'] + 1e-8)).mean():.2f}x
    """
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig(output_path / f'bound_tightness_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved tightness analysis to {output_path / f'bound_tightness_layer{layer_idx}.png'}")


def analyze_pruning_potential(layer_idx: int = 14, device: str = 'cuda'):
    """
    Analyze what pruning ratio we could achieve with tighter bounds.
    """
    layer_path = DATA_ROOT / f"layer_{layer_idx}"

    q = torch.load(layer_path / "q_rope.pt", weights_only=True, map_location='cpu')
    k = torch.load(layer_path / "k_rope.pt", weights_only=True, map_location='cpu')

    q = q.float().to(device)
    k = k.float().to(device)

    B, HQ, T, K_dim = q.shape
    _, HKV, _, _ = k.shape
    G = HQ // HKV

    block_size = 128
    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)
    delta = 5.0

    q_single = q[:, :, -1, :]  # [B, HQ, K]

    # Collect all block scores
    all_fp_scores = []  # [HQ, num_blocks]
    all_minmax_bounds = []
    all_norm_bounds = []

    for hq in range(HQ):
        hkv = hq // G
        q_vec = q_single[:, hq, :]  # [B, K]
        q_norm = q_vec.norm(dim=-1)

        fp_scores_head = []
        minmax_head = []
        norm_head = []

        for blk_idx in range(num_blocks):
            start = blk_idx * block_size
            end = min(start + block_size, T)

            k_blk = k[:, hkv, start:end, :]  # [B, blk_len, K]

            # FP score
            scores = torch.einsum('bk,btk->bt', q_vec, k_blk) * scale
            fp_max = scores.max(dim=-1).values.item()

            # MinMax bound
            k_min = k_blk.amin(dim=1)
            k_max = k_blk.amax(dim=1)
            k_opt = torch.where(q_vec > 0, k_max, k_min)
            minmax_bound = (q_vec * k_opt).sum(dim=-1) * scale

            # Norm bound
            k_norms = k_blk.norm(dim=-1)
            k_norm_max = k_norms.amax(dim=-1)
            norm_bound = q_norm * k_norm_max * scale

            fp_scores_head.append(fp_max)
            minmax_head.append(minmax_bound.item())
            norm_head.append(norm_bound.item())

        all_fp_scores.append(fp_scores_head)
        all_minmax_bounds.append(minmax_head)
        all_norm_bounds.append(norm_head)

    all_fp_scores = np.array(all_fp_scores)  # [HQ, num_blocks]
    all_minmax_bounds = np.array(all_minmax_bounds)
    all_norm_bounds = np.array(all_norm_bounds)

    # Compute thresholds per head (max of first and last block - delta)
    fp_threshold = np.maximum(all_fp_scores[:, 0], all_fp_scores[:, -1]) - delta  # [HQ]

    # Compute pruning masks
    fp_prune = all_fp_scores < fp_threshold[:, None]
    minmax_prune = all_minmax_bounds < fp_threshold[:, None]
    norm_prune = all_norm_bounds < fp_threshold[:, None]

    # Statistics
    print(f"\nPruning Analysis for Layer {layer_idx}:")
    print(f"  Threshold range: [{fp_threshold.min():.2f}, {fp_threshold.max():.2f}]")
    print(f"  FP prune ratio: {fp_prune.mean()*100:.1f}%")
    print(f"  MinMax prune ratio: {minmax_prune.mean()*100:.1f}%")
    print(f"  Norm prune ratio: {norm_prune.mean()*100:.1f}%")

    # What if we use tighter bounds?
    # Simulate "perfect" pruning where we use actual FP max score
    # Compare with bounds

    # Compute required "tightening factor" to match FP pruning
    # For MinMax: we need minmax_bound * factor < threshold
    # For blocks that should be pruned (fp < threshold), find minimum factor

    for name, bounds in [('MinMax', all_minmax_bounds), ('Norm', all_norm_bounds)]:
        # For blocks that FP would prune (fp < threshold)
        fp_would_prune = all_fp_scores < fp_threshold[:, None]

        if fp_would_prune.any():
            # Compute how much we need to tighten the bound
            # For these blocks: bound * factor = fp_score
            # factor = fp_score / bound
            factors = all_fp_scores[fp_would_prune] / (bounds[fp_would_prune] + 1e-8)
            print(f"  {name} tightening factor needed: mean={factors.mean():.3f}, min={factors.min():.3f}")

    return {
        'fp_scores': all_fp_scores,
        'minmax_bounds': all_minmax_bounds,
        'norm_bounds': all_norm_bounds,
        'threshold': fp_threshold,
        'fp_prune': fp_prune,
        'minmax_prune': minmax_prune,
        'norm_prune': norm_prune,
    }


def visualize_pruning_potential(results: Dict, output_path: Path, layer_idx: int):
    """
    Visualize pruning potential analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    HQ, num_blocks = results['fp_scores'].shape

    # 1. Block scores heatmap for different methods
    ax1 = axes[0, 0]
    im = ax1.imshow(results['fp_scores'], aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax1)
    ax1.set_xlabel('Block Index')
    ax1.set_ylabel('Head Index')
    ax1.set_title('FP Block Max Scores')

    # 2. Threshold comparison
    ax2 = axes[0, 1]
    head_idx = 0  # Show one head
    blocks = np.arange(num_blocks)
    ax2.plot(blocks, results['fp_scores'][head_idx], 'b-', label='FP', alpha=0.7)
    ax2.plot(blocks, results['minmax_bounds'][head_idx], 'orange', linestyle='--', label='MinMax UB', alpha=0.7)
    ax2.plot(blocks, results['norm_bounds'][head_idx], 'r:', label='Norm UB', alpha=0.7)
    ax2.axhline(y=results['threshold'][head_idx], color='k', linestyle='--', label='Threshold')
    ax2.fill_between(blocks, results['threshold'][head_idx], results['fp_scores'][head_idx],
                     where=results['fp_prune'][head_idx], alpha=0.3, color='blue', label='FP Prune')
    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Head {head_idx} Score Comparison')
    ax2.legend(fontsize=8)

    # 3. Bound ratio distribution
    ax3 = axes[1, 0]
    minmax_ratio = results['minmax_bounds'].flatten() / (results['fp_scores'].flatten() + 1e-8)
    norm_ratio = results['norm_bounds'].flatten() / (results['fp_scores'].flatten() + 1e-8)
    ax3.hist(minmax_ratio, bins=50, alpha=0.5, label=f'MinMax (mean={minmax_ratio.mean():.2f}x)', color='orange')
    ax3.hist(norm_ratio, bins=50, alpha=0.5, label=f'Norm (mean={norm_ratio.mean():.2f}x)', color='red')
    ax3.axvline(x=1.0, color='k', linestyle='--', label='Tight (1.0x)')
    ax3.set_xlabel('Upper Bound / FP Score Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Bound Looseness Distribution')
    ax3.legend()
    ax3.set_xlim([0, min(10, norm_ratio.max())])

    # 4. Prune mask comparison
    ax4 = axes[1, 1]
    prune_data = np.stack([
        results['fp_prune'].astype(float),
        results['minmax_prune'].astype(float),
        results['norm_prune'].astype(float),
    ])  # [3, HQ, num_blocks]
    # Subsample for visualization
    max_show = 100
    if num_blocks > max_show:
        step = num_blocks // max_show
        prune_data = prune_data[:, :, ::step]

    # Show average across heads
    prune_avg = prune_data.mean(axis=1)  # [3, blocks]
    ax4.imshow(prune_avg, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['FP', 'MinMax', 'Norm'])
    ax4.set_xlabel('Block Index (subsampled)')
    ax4.set_title('Prune Mask Comparison (avg over heads)\nGreen=Keep, Red=Prune')

    plt.tight_layout()
    plt.savefig(output_path / f'pruning_potential_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved pruning potential to {output_path / f'pruning_potential_layer{layer_idx}.png'}")


def main():
    print("=" * 60)
    print("Bound Tightness Deep Analysis")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Analyze a middle layer
    layer_idx = 14

    print(f"\n1. Analyzing bound tightness for layer {layer_idx}...")
    tightness_results = analyze_bound_tightness(layer_idx, device)
    visualize_tightness(tightness_results, OUTPUT_DIR, layer_idx)

    print(f"\n2. Analyzing pruning potential for layer {layer_idx}...")
    pruning_results = analyze_pruning_potential(layer_idx, device)
    visualize_pruning_potential(pruning_results, OUTPUT_DIR, layer_idx)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)

    minmax_ratio = tightness_results['minmax_bounds'] / (tightness_results['fp_scores'] + 1e-8)
    norm_ratio = tightness_results['norm_bounds'] / (tightness_results['fp_scores'] + 1e-8)

    print(f"""
    The coarse-grained bounds are too loose for effective pruning:

    1. MinMax Bound Analysis:
       - Mean gap to FP: {tightness_results['gap_minmax'].mean():.2f}
       - Bound/FP ratio: {minmax_ratio.mean():.2f}x (ideal: 1.0x)

    2. Norm Bound Analysis:
       - Mean gap to FP: {tightness_results['gap_norm'].mean():.2f}
       - Bound/FP ratio: {norm_ratio.mean():.2f}x (ideal: 1.0x)

    3. Why bounds are loose:
       - MinMax: Assumes each dimension can independently take its optimal value
       - Norm: Uses max norm, but max(q·k) << ||q||·||k||_max due to alignment

    4. INT2 quantization works because:
       - It computes actual q·k_quantized, not an upper bound
       - Quantization error is small, preserving relative ordering
       - Delta threshold accounts for quantization noise

    Recommendation: Coarse bounds (MinMax/Norm) cannot replace INT2 filtering
    because they produce bounds that are {minmax_ratio.mean():.1f}x-{norm_ratio.mean():.1f}x
    looser than actual scores, making pruning ineffective.
    """)


if __name__ == "__main__":
    main()

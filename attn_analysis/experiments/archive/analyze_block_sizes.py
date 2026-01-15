"""
Optimized analysis of block size effects on coarse-grained filtering.
Uses vectorized operations for faster computation.
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
OUTPUT_BASE = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/output_blocksize")


@torch.no_grad()
def analyze_layer_vectorized(
    layer_idx: int,
    block_size: int,
    delta: float = 5.0,
    device: str = 'cuda',
) -> Dict:
    """
    Analyze a single layer with vectorized operations.
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

    # Pad k to make it evenly divisible by block_size
    pad_len = num_blocks * block_size - T
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len), value=0)

    # Reshape k into blocks: [B, HKV, num_blocks, block_size, K]
    k_blocks = k.view(B, HKV, num_blocks, block_size, K_dim)

    # Compute per-block statistics
    k_min = k_blocks.amin(dim=3)  # [B, HKV, num_blocks, K]
    k_max = k_blocks.amax(dim=3)
    k_norms = k_blocks.norm(dim=-1)  # [B, HKV, num_blocks, block_size]
    k_norm_max = k_norms.amax(dim=-1)  # [B, HKV, num_blocks]

    # Initialize result arrays
    all_fp_scores = torch.zeros(HQ, num_blocks, device=device)
    all_minmax_bounds = torch.zeros(HQ, num_blocks, device=device)
    all_norm_bounds = torch.zeros(HQ, num_blocks, device=device)

    for hkv in range(HKV):
        # Get k blocks for this KV head: [B, num_blocks, block_size, K]
        k_hkv = k_blocks[:, hkv]

        for g in range(G):
            hq = hkv * G + g
            q_vec = q_single[:, hq, :]  # [B, K]
            q_norm = q_vec.norm(dim=-1)  # [B]

            # FP scores: q @ k^T for each block
            # q_vec: [B, K], k_hkv: [B, num_blocks, block_size, K]
            # scores: [B, num_blocks, block_size]
            scores = torch.einsum('bk,bntk->bnt', q_vec, k_hkv) * scale
            fp_max = scores.amax(dim=-1)  # [B, num_blocks]

            # MinMax bound
            # k_opt[d] = k_max[d] if q[d] > 0 else k_min[d]
            k_opt = torch.where(
                q_vec[:, None, :] > 0,
                k_max[:, hkv],  # [B, num_blocks, K]
                k_min[:, hkv]
            )
            minmax_bound = (q_vec[:, None, :] * k_opt).sum(dim=-1) * scale  # [B, num_blocks]

            # Norm bound
            norm_bound = q_norm[:, None] * k_norm_max[:, hkv] * scale  # [B, num_blocks]

            all_fp_scores[hq] = fp_max[0]
            all_minmax_bounds[hq] = minmax_bound[0]
            all_norm_bounds[hq] = norm_bound[0]

    # Convert to numpy
    all_fp_scores = all_fp_scores.cpu().numpy()
    all_minmax_bounds = all_minmax_bounds.cpu().numpy()
    all_norm_bounds = all_norm_bounds.cpu().numpy()

    # Compute thresholds
    fp_threshold = np.maximum(all_fp_scores[:, 0], all_fp_scores[:, -1]) - delta

    # Compute pruning masks
    fp_prune = all_fp_scores < fp_threshold[:, None]
    minmax_prune = all_minmax_bounds < fp_threshold[:, None]
    norm_prune = all_norm_bounds < fp_threshold[:, None]

    # Compute gaps
    minmax_gap = all_minmax_bounds - all_fp_scores
    norm_gap = all_norm_bounds - all_fp_scores

    return {
        'layer_idx': layer_idx,
        'block_size': block_size,
        'num_blocks': num_blocks,
        'fp_scores': all_fp_scores,
        'minmax_bounds': all_minmax_bounds,
        'norm_bounds': all_norm_bounds,
        'threshold': fp_threshold,
        'fp_prune_ratio': fp_prune.mean(),
        'minmax_prune_ratio': minmax_prune.mean(),
        'norm_prune_ratio': norm_prune.mean(),
        'minmax_gap_mean': minmax_gap.mean(),
        'norm_gap_mean': norm_gap.mean(),
    }


def analyze_all_block_sizes(
    block_sizes: List[int],
    layers: List[int],
    delta: float = 5.0,
    device: str = 'cuda',
) -> Dict[int, List[Dict]]:
    """
    Analyze all layers for each block size.
    """
    results = {}

    for bs in block_sizes:
        print(f"\n{'='*60}")
        print(f"Analyzing block size: {bs}")
        print(f"{'='*60}")

        output_dir = OUTPUT_BASE / f"bs_{bs}"
        output_dir.mkdir(parents=True, exist_ok=True)

        bs_results = []
        for layer_idx in tqdm(layers, desc=f"BS={bs}"):
            try:
                result = analyze_layer_vectorized(layer_idx, bs, delta, device)
                bs_results.append(result)
            except Exception as e:
                print(f"Error at layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()

        results[bs] = bs_results
        torch.save(bs_results, output_dir / 'results.pt')

    return results


def visualize_blocksize_comparison(results: Dict[int, List[Dict]], output_dir: Path):
    """
    Create comparison visualizations across different block sizes.
    """
    block_sizes = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract average metrics
    avg_fp = []
    avg_minmax = []
    avg_norm = []
    avg_minmax_gap = []
    avg_norm_gap = []

    for bs in block_sizes:
        avg_fp.append(np.mean([r['fp_prune_ratio'] * 100 for r in results[bs]]))
        avg_minmax.append(np.mean([r['minmax_prune_ratio'] * 100 for r in results[bs]]))
        avg_norm.append(np.mean([r['norm_prune_ratio'] * 100 for r in results[bs]]))
        avg_minmax_gap.append(np.mean([r['minmax_gap_mean'] for r in results[bs]]))
        avg_norm_gap.append(np.mean([r['norm_gap_mean'] for r in results[bs]]))

    # 1. Prune ratio vs block size
    ax1 = axes[0, 0]
    ax1.plot(block_sizes, avg_fp, 'bo-', label='FP (GT)', linewidth=2, markersize=8)
    ax1.plot(block_sizes, avg_minmax, 'o-', color='orange', label='MinMax', linewidth=2, markersize=8)
    ax1.plot(block_sizes, avg_norm, 'rs-', label='Norm', linewidth=2, markersize=8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Average Prune Ratio (%)')
    ax1.set_title('Prune Ratio vs Block Size')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log', base=2)

    # 2. Bound gap vs block size
    ax2 = axes[0, 1]
    ax2.plot(block_sizes, avg_minmax_gap, 'o-', color='orange', label='MinMax Gap', linewidth=2, markersize=8)
    ax2.plot(block_sizes, avg_norm_gap, 'rs-', label='Norm Gap', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Mean Gap (Upper Bound - FP)')
    ax2.set_title('Bound Tightness vs Block Size')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log', base=2)

    # 3. Bar chart comparison
    ax3 = axes[1, 0]
    x = np.arange(len(block_sizes))
    width = 0.25
    ax3.bar(x - width, avg_fp, width, label='FP (GT)', color='blue', alpha=0.7)
    ax3.bar(x, avg_minmax, width, label='MinMax', color='orange', alpha=0.7)
    ax3.bar(x + width, avg_norm, width, label='Norm', color='red', alpha=0.7)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Average Prune Ratio (%)')
    ax3.set_title('Average Prune Ratio by Block Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels(block_sizes)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Per-layer comparison for different block sizes
    ax4 = axes[1, 1]
    layers = [r['layer_idx'] for r in results[block_sizes[0]]]

    for bs in [block_sizes[0], block_sizes[len(block_sizes)//2], block_sizes[-1]]:
        minmax_prune = [r['minmax_prune_ratio'] * 100 for r in results[bs]]
        ax4.plot(layers, minmax_prune, 'o-', label=f'MinMax BS={bs}', markersize=3)

    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('MinMax Prune Ratio (%)')
    ax4.set_title('MinMax Prune Ratio by Layer (Selected Block Sizes)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'blocksize_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison to {output_dir / 'blocksize_comparison.png'}")


def visualize_per_blocksize(results: Dict[int, List[Dict]]):
    """
    Create detailed visualizations for each block size.
    """
    for bs, bs_results in results.items():
        output_dir = OUTPUT_BASE / f"bs_{bs}"
        output_dir.mkdir(parents=True, exist_ok=True)

        layers = [r['layer_idx'] for r in bs_results]
        fp_prune = [r['fp_prune_ratio'] * 100 for r in bs_results]
        minmax_prune = [r['minmax_prune_ratio'] * 100 for r in bs_results]
        norm_prune = [r['norm_prune_ratio'] * 100 for r in bs_results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Block Size = {bs}', fontsize=14)

        # 1. Prune ratio by layer
        ax1 = axes[0, 0]
        x = np.arange(len(layers))
        width = 0.25
        ax1.bar(x - width, fp_prune, width, label='FP (GT)', color='blue', alpha=0.7)
        ax1.bar(x, minmax_prune, width, label='MinMax', color='orange', alpha=0.7)
        ax1.bar(x + width, norm_prune, width, label='Norm', color='red', alpha=0.7)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Prune Ratio (%)')
        ax1.set_title('Prune Ratio by Method')
        ax1.set_xticks(x[::4])
        ax1.set_xticklabels(layers[::4])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Score distribution for middle layer
        mid_idx = len(bs_results) // 2
        r = bs_results[mid_idx]

        ax2 = axes[0, 1]
        fp_flat = r['fp_scores'].flatten()
        minmax_flat = r['minmax_bounds'].flatten()

        ax2.hist(fp_flat, bins=50, alpha=0.5, label='FP', color='blue', density=True)
        ax2.hist(minmax_flat, bins=50, alpha=0.5, label='MinMax UB', color='orange', density=True)
        ax2.axvline(x=r['threshold'].mean(), color='k', linestyle='--', label='Avg Threshold')
        ax2.set_xlabel('Attention Score')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Layer {r["layer_idx"]} Score Distribution')
        ax2.legend()

        # 3. Block scores for one head
        ax3 = axes[1, 0]
        head_idx = 0
        num_show = min(500, r['num_blocks'])
        step = max(1, r['num_blocks'] // num_show)
        blocks = np.arange(0, r['num_blocks'], step)

        ax3.plot(blocks, r['fp_scores'][head_idx, ::step], 'b-', label='FP', alpha=0.7, linewidth=1)
        ax3.plot(blocks, r['minmax_bounds'][head_idx, ::step], color='orange', linestyle='--', label='MinMax UB', alpha=0.7)
        ax3.axhline(y=r['threshold'][head_idx], color='k', linestyle='--', label='Threshold')
        ax3.set_xlabel('Block Index')
        ax3.set_ylabel('Score')
        ax3.set_title(f'Layer {r["layer_idx"]} Head {head_idx} Scores')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # 4. Summary stats
        ax4 = axes[1, 1]
        stats_text = f"""Block Size = {bs}
Num Blocks: {r['num_blocks']}

Average Prune Ratios:
  FP (Ground Truth): {np.mean(fp_prune):.1f}%
  MinMax Coarse:     {np.mean(minmax_prune):.1f}%
  Norm Coarse:       {np.mean(norm_prune):.1f}%

Average Bound Gaps:
  MinMax: {np.mean([r['minmax_gap_mean'] for r in bs_results]):.2f}
  Norm:   {np.mean([r['norm_gap_mean'] for r in bs_results]):.2f}

Effectiveness:
  MinMax captures {np.mean(minmax_prune)/(np.mean(fp_prune)+1e-8)*100:.1f}% of FP pruning
  Norm captures   {np.mean(norm_prune)/(np.mean(fp_prune)+1e-8)*100:.1f}% of FP pruning
"""
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'detailed_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved detailed analysis to {output_dir / 'detailed_analysis.png'}")


def create_summary_table(results: Dict[int, List[Dict]], output_dir: Path):
    """
    Create a summary table comparing all block sizes.
    """
    block_sizes = sorted(results.keys())

    with open(output_dir / 'summary_all_blocksizes.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Block Size Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'BS':>8} {'#Blocks':>8} {'FP%':>8} {'MinMax%':>10} {'Norm%':>8} {'MM Gap':>10} {'N Gap':>10}\n")
        f.write("-" * 80 + "\n")

        for bs in block_sizes:
            bs_results = results[bs]
            num_blocks = bs_results[0]['num_blocks']
            fp_prune = np.mean([r['fp_prune_ratio'] * 100 for r in bs_results])
            minmax_prune = np.mean([r['minmax_prune_ratio'] * 100 for r in bs_results])
            norm_prune = np.mean([r['norm_prune_ratio'] * 100 for r in bs_results])
            minmax_gap = np.mean([r['minmax_gap_mean'] for r in bs_results])
            norm_gap = np.mean([r['norm_gap_mean'] for r in bs_results])

            f.write(f"{bs:>8} {num_blocks:>8} {fp_prune:>8.1f} {minmax_prune:>10.1f} {norm_prune:>8.1f} {minmax_gap:>10.2f} {norm_gap:>10.2f}\n")

        f.write("-" * 80 + "\n")

    print(f"Saved summary to {output_dir / 'summary_all_blocksizes.txt'}")


def main():
    print("=" * 60)
    print("Block Size Effect Analysis (Optimized)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test different block sizes
    block_sizes = [8, 16, 32, 64, 128, 256]

    # Get available layers
    available_layers = sorted([int(p.name.split('_')[1]) for p in DATA_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(available_layers)} layers")

    # Analyze all block sizes
    results = analyze_all_block_sizes(
        block_sizes=block_sizes,
        layers=available_layers,
        delta=5.0,
        device=device,
    )

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_blocksize_comparison(results, OUTPUT_BASE)
    visualize_per_blocksize(results)
    create_summary_table(results, OUTPUT_BASE)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for bs in block_sizes:
        bs_results = results[bs]
        fp_avg = np.mean([r['fp_prune_ratio'] * 100 for r in bs_results])
        minmax_avg = np.mean([r['minmax_prune_ratio'] * 100 for r in bs_results])
        norm_avg = np.mean([r['norm_prune_ratio'] * 100 for r in bs_results])
        minmax_gap = np.mean([r['minmax_gap_mean'] for r in bs_results])
        norm_gap = np.mean([r['norm_gap_mean'] for r in bs_results])

        print(f"\nBlock Size = {bs}:")
        print(f"  FP Prune:     {fp_avg:.1f}%")
        print(f"  MinMax Prune: {minmax_avg:.1f}% (gap: {minmax_gap:.2f})")
        print(f"  Norm Prune:   {norm_avg:.1f}% (gap: {norm_gap:.2f})")


if __name__ == "__main__":
    main()

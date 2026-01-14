"""
Compare different block filtering methods for sparse attention:
1. Original INT2 quantization based filtering
2. Coarse-grained filtering using K norm max and per-dimension min/max bounds

This script analyzes how these methods compare in terms of:
- Number of blocks pruned
- Overlap between pruned blocks
- Attention score distributions
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def symmetric_int2_quantize(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric INT2 quantization for K.
    k: [B, HKV, T, K]
    Returns:
        k_q: quantized values (0, 1, 2, 3)
        k_scale: per-channel scale [B, HKV, K]
    """
    B, HKV, T, K_dim = k.shape
    # Compute per-channel scale (max abs value across time dimension)
    k_abs_max = k.abs().amax(dim=2, keepdim=True)  # [B, HKV, 1, K]
    QMAX = 3  # 2-bit: 0, 1, 2, 3
    QZERO = QMAX / 2.0  # 1.5 for symmetric

    # Scale: maps [-max, max] to [0, 3]
    k_scale = k_abs_max.squeeze(2) / QZERO  # [B, HKV, K]
    k_scale = k_scale.clamp(min=1e-8)  # avoid division by zero

    # Quantize: k_q = round(k / scale + QZERO)
    k_normalized = k / k_scale.unsqueeze(2)  # [B, HKV, T, K]
    k_q = (k_normalized + QZERO).round().clamp(0, QMAX).to(torch.int8)

    return k_q, k_scale


def compute_int2_block_scores(
    q: torch.Tensor,  # [B, HQ, 1, K] - single query
    k_q: torch.Tensor,  # [B, HKV, T, K] - quantized keys
    k_scale: torch.Tensor,  # [B, HKV, K] - per-channel scale
    block_size: int = 128,
) -> torch.Tensor:
    """
    Compute attention scores per block using INT2 quantized keys.
    Returns block max scores: [B, HQ, num_blocks]
    """
    B, HQ, _, K_dim = q.shape
    _, HKV, T, _ = k_q.shape
    G = HQ // HKV  # Group size for GQA

    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)
    QZERO = 1.5  # INT2 symmetric zero point

    block_max_scores = torch.zeros(B, HQ, num_blocks, device=q.device, dtype=torch.float32)

    for blk_idx in range(num_blocks):
        start = blk_idx * block_size
        end = min(start + block_size, T)

        k_q_blk = k_q[:, :, start:end, :].float()  # [B, HKV, blk_len, K]

        for hkv in range(HKV):
            # Get scale for this KV head
            s = k_scale[:, hkv, :]  # [B, K]

            # Dequantize: k_dequant = (k_q - QZERO) * scale
            k_dequant = (k_q_blk[:, hkv, :, :] - QZERO) * s.unsqueeze(1)  # [B, blk_len, K]

            # Compute attention scores for all Q heads in this group
            for g in range(G):
                hq = hkv * G + g
                q_vec = q[:, hq, 0, :]  # [B, K]

                # scores = q @ k^T * scale
                scores = torch.einsum('bk,btk->bt', q_vec, k_dequant) * scale
                block_max_scores[:, hq, blk_idx] = scores.max(dim=-1).values

    return block_max_scores


def compute_fp_block_scores(
    q: torch.Tensor,  # [B, HQ, 1, K]
    k: torch.Tensor,  # [B, HKV, T, K]
    block_size: int = 128,
) -> torch.Tensor:
    """
    Compute attention scores per block using full-precision keys.
    Returns block max scores: [B, HQ, num_blocks]
    """
    B, HQ, _, K_dim = q.shape
    _, HKV, T, _ = k.shape
    G = HQ // HKV

    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)

    block_max_scores = torch.zeros(B, HQ, num_blocks, device=q.device, dtype=torch.float32)

    for blk_idx in range(num_blocks):
        start = blk_idx * block_size
        end = min(start + block_size, T)

        k_blk = k[:, :, start:end, :].float()  # [B, HKV, blk_len, K]

        for hkv in range(HKV):
            for g in range(G):
                hq = hkv * G + g
                q_vec = q[:, hq, 0, :].float()  # [B, K]

                scores = torch.einsum('bk,btk->bt', q_vec, k_blk[:, hkv]) * scale
                block_max_scores[:, hq, blk_idx] = scores.max(dim=-1).values

    return block_max_scores


def compute_coarse_filter_bounds(
    q: torch.Tensor,  # [B, HQ, 1, K]
    k: torch.Tensor,  # [B, HKV, T, K]
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute coarse-grained upper bound for block attention scores.

    Method: For each block, compute upper bound using:
    - K norm max: max L2 norm of K vectors in the block
    - Per-dimension min/max: For each dim, take min and max values

    Upper bound for q·k:
    - If q[d] > 0: use k_max[d]
    - If q[d] < 0: use k_min[d]
    This gives the maximum possible dot product.

    Returns:
        upper_bound_scores: [B, HQ, num_blocks] - upper bound scores
        norm_based_scores: [B, HQ, num_blocks] - norm-based upper bound (||q|| * ||k||_max)
    """
    B, HQ, _, K_dim = q.shape
    _, HKV, T, _ = k.shape
    G = HQ // HKV

    num_blocks = (T + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(K_dim)

    upper_bound_scores = torch.zeros(B, HQ, num_blocks, device=q.device, dtype=torch.float32)
    norm_based_scores = torch.zeros(B, HQ, num_blocks, device=q.device, dtype=torch.float32)

    for blk_idx in range(num_blocks):
        start = blk_idx * block_size
        end = min(start + block_size, T)

        k_blk = k[:, :, start:end, :].float()  # [B, HKV, blk_len, K]

        # Per-dimension min/max for each KV head
        k_min = k_blk.amin(dim=2)  # [B, HKV, K]
        k_max = k_blk.amax(dim=2)  # [B, HKV, K]

        # Norm-based upper bound: ||k||_max per block
        k_norms = k_blk.norm(dim=-1)  # [B, HKV, blk_len]
        k_norm_max = k_norms.amax(dim=-1)  # [B, HKV]

        for hkv in range(HKV):
            for g in range(G):
                hq = hkv * G + g
                q_vec = q[:, hq, 0, :].float()  # [B, K]

                # Min-Max based upper bound:
                # For each dimension, choose k_min or k_max based on sign of q
                k_opt = torch.where(q_vec > 0, k_max[:, hkv], k_min[:, hkv])  # [B, K]
                upper_score = (q_vec * k_opt).sum(dim=-1) * scale  # [B]
                upper_bound_scores[:, hq, blk_idx] = upper_score

                # Norm-based upper bound: ||q|| * ||k||_max * scale
                q_norm = q_vec.norm(dim=-1)  # [B]
                norm_score = q_norm * k_norm_max[:, hkv] * scale
                norm_based_scores[:, hq, blk_idx] = norm_score

    return upper_bound_scores, norm_based_scores


def compute_threshold_and_mask(
    block_scores: torch.Tensor,  # [B, HQ, num_blocks]
    delta: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute threshold using first and last block, then create pruning mask.
    Returns:
        threshold: [B, HQ]
        mask: [B, HQ, num_blocks] - True means KEEP, False means PRUNE
    """
    B, HQ, num_blocks = block_scores.shape

    # Threshold = max(first_block_score, last_block_score) - delta
    first_scores = block_scores[:, :, 0]  # [B, HQ]
    last_scores = block_scores[:, :, -1]  # [B, HQ]
    threshold = torch.maximum(first_scores, last_scores) - delta

    # Mask: keep blocks where max_score >= threshold
    mask = block_scores >= threshold.unsqueeze(-1)

    return threshold, mask


def analyze_layer(
    layer_idx: int,
    query_pos: int = -1,  # Position of query (-1 means last)
    block_size: int = 128,
    delta: float = 5.0,
    device: str = 'cuda',
) -> Dict:
    """
    Analyze a single layer comparing different filtering methods.
    """
    layer_path = DATA_ROOT / f"layer_{layer_idx}"

    # Load data with map_location to handle multi-GPU saved tensors
    q = torch.load(layer_path / "q_rope.pt", weights_only=True, map_location='cpu')  # [B, HQ, T, K]
    k = torch.load(layer_path / "k_rope.pt", weights_only=True, map_location='cpu')  # [B, HKV, T, K]

    # Convert to float32 for computation
    q = q.float().to(device)
    k = k.float().to(device)

    B, HQ, T, K_dim = q.shape
    _, HKV, _, _ = k.shape

    # Take query at specific position
    q_single = q[:, :, query_pos:query_pos+1, :] if query_pos != -1 else q[:, :, -1:, :]

    # 1. INT2 quantization based filtering
    k_q, k_scale = symmetric_int2_quantize(k)
    int2_block_scores = compute_int2_block_scores(q_single, k_q, k_scale, block_size)
    int2_threshold, int2_mask = compute_threshold_and_mask(int2_block_scores, delta)

    # 2. Full precision block scores (ground truth)
    fp_block_scores = compute_fp_block_scores(q_single, k, block_size)
    fp_threshold, fp_mask = compute_threshold_and_mask(fp_block_scores, delta)

    # 3. Coarse-grained filtering (min-max bounds)
    upper_bound_scores, norm_based_scores = compute_coarse_filter_bounds(q_single, k, block_size)

    # For coarse filtering, we use upper bound: if upper_bound < threshold, prune
    # This is more conservative (keeps more blocks)
    coarse_threshold_minmax, _ = compute_threshold_and_mask(fp_block_scores, delta)  # Use FP threshold
    coarse_mask_minmax = upper_bound_scores >= coarse_threshold_minmax.unsqueeze(-1)

    coarse_threshold_norm, _ = compute_threshold_and_mask(fp_block_scores, delta)
    coarse_mask_norm = norm_based_scores >= coarse_threshold_norm.unsqueeze(-1)

    # Compute statistics
    num_blocks = int2_mask.shape[-1]

    results = {
        'layer_idx': layer_idx,
        'num_blocks': num_blocks,
        'T': T,
        'HQ': HQ,
        'HKV': HKV,
        'K_dim': K_dim,
        'block_size': block_size,
        'delta': delta,

        # Block scores
        'fp_block_scores': fp_block_scores.cpu(),
        'int2_block_scores': int2_block_scores.cpu(),
        'upper_bound_scores': upper_bound_scores.cpu(),
        'norm_based_scores': norm_based_scores.cpu(),

        # Thresholds
        'fp_threshold': fp_threshold.cpu(),
        'int2_threshold': int2_threshold.cpu(),

        # Masks (True = keep)
        'fp_mask': fp_mask.cpu(),
        'int2_mask': int2_mask.cpu(),
        'coarse_mask_minmax': coarse_mask_minmax.cpu(),
        'coarse_mask_norm': coarse_mask_norm.cpu(),

        # Prune ratios
        'fp_prune_ratio': (~fp_mask).float().mean().item(),
        'int2_prune_ratio': (~int2_mask).float().mean().item(),
        'coarse_minmax_prune_ratio': (~coarse_mask_minmax).float().mean().item(),
        'coarse_norm_prune_ratio': (~coarse_mask_norm).float().mean().item(),
    }

    return results


def visualize_comparison(results_list: List[Dict], output_path: Path):
    """
    Create visualizations comparing filtering methods across layers.
    """
    num_layers = len(results_list)

    # Extract data
    layers = [r['layer_idx'] for r in results_list]
    fp_prune = [r['fp_prune_ratio'] * 100 for r in results_list]
    int2_prune = [r['int2_prune_ratio'] * 100 for r in results_list]
    minmax_prune = [r['coarse_minmax_prune_ratio'] * 100 for r in results_list]
    norm_prune = [r['coarse_norm_prune_ratio'] * 100 for r in results_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Prune ratio comparison across layers
    ax1 = axes[0, 0]
    x = np.arange(num_layers)
    width = 0.2
    ax1.bar(x - 1.5*width, fp_prune, width, label='FP (Ground Truth)', color='blue', alpha=0.7)
    ax1.bar(x - 0.5*width, int2_prune, width, label='INT2 Quantized', color='green', alpha=0.7)
    ax1.bar(x + 0.5*width, minmax_prune, width, label='Coarse (Min-Max)', color='orange', alpha=0.7)
    ax1.bar(x + 1.5*width, norm_prune, width, label='Coarse (Norm)', color='red', alpha=0.7)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Prune Ratio (%)')
    ax1.set_title('Block Prune Ratio by Filtering Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Per-layer detailed comparison for a specific layer (middle layer)
    mid_layer_idx = num_layers // 2
    r = results_list[mid_layer_idx]
    ax2 = axes[0, 1]

    # Plot block scores for one head
    head_idx = 0
    fp_scores = r['fp_block_scores'][0, head_idx].numpy()
    int2_scores = r['int2_block_scores'][0, head_idx].numpy()
    upper_scores = r['upper_bound_scores'][0, head_idx].numpy()
    norm_scores = r['norm_based_scores'][0, head_idx].numpy()
    threshold = r['fp_threshold'][0, head_idx].item()

    block_indices = np.arange(len(fp_scores))
    ax2.plot(block_indices, fp_scores, 'b-', label='FP Scores', alpha=0.7, linewidth=1)
    ax2.plot(block_indices, int2_scores, 'g--', label='INT2 Scores', alpha=0.7, linewidth=1)
    ax2.plot(block_indices, upper_scores, 'orange', linestyle=':', label='MinMax Upper Bound', alpha=0.7, linewidth=1.5)
    ax2.plot(block_indices, norm_scores, 'r:', label='Norm Upper Bound', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=threshold, color='k', linestyle='--', label=f'Threshold (δ={r["delta"]})')
    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Max Attention Score')
    ax2.set_title(f'Layer {r["layer_idx"]} Head {head_idx} Block Scores')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Mask overlap analysis (INT2 vs Coarse methods)
    ax3 = axes[1, 0]

    # Compute overlap metrics per layer
    int2_vs_minmax_agree = []
    int2_vs_norm_agree = []
    int2_catches_fp_prune = []  # INT2 prunes blocks that FP would prune
    minmax_catches_fp_prune = []

    for r in results_list:
        int2_m = r['int2_mask'][0].float().mean(dim=0)  # [num_blocks]
        minmax_m = r['coarse_mask_minmax'][0].float().mean(dim=0)
        norm_m = r['coarse_mask_norm'][0].float().mean(dim=0)
        fp_m = r['fp_mask'][0].float().mean(dim=0)

        # Agreement rate
        int2_vs_minmax_agree.append(((int2_m > 0.5) == (minmax_m > 0.5)).float().mean().item() * 100)
        int2_vs_norm_agree.append(((int2_m > 0.5) == (norm_m > 0.5)).float().mean().item() * 100)

        # How well does INT2/coarse catch what FP prunes?
        fp_prune_mask = fp_m < 0.5  # Blocks FP prunes
        if fp_prune_mask.any():
            int2_catches = ((int2_m < 0.5) & fp_prune_mask).sum() / fp_prune_mask.sum() * 100
            minmax_catches = ((minmax_m < 0.5) & fp_prune_mask).sum() / fp_prune_mask.sum() * 100
        else:
            int2_catches = 100.0
            minmax_catches = 100.0
        int2_catches_fp_prune.append(int2_catches.item() if torch.is_tensor(int2_catches) else int2_catches)
        minmax_catches_fp_prune.append(minmax_catches.item() if torch.is_tensor(minmax_catches) else minmax_catches)

    ax3.plot(layers, int2_vs_minmax_agree, 'o-', label='INT2 vs MinMax Agreement', color='purple')
    ax3.plot(layers, int2_vs_norm_agree, 's-', label='INT2 vs Norm Agreement', color='brown')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Agreement Rate (%)')
    ax3.set_title('Mask Agreement Between Methods')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 105])

    # 4. Score distribution comparison (histogram for middle layer)
    ax4 = axes[1, 1]
    r = results_list[mid_layer_idx]
    fp_all = r['fp_block_scores'].flatten().numpy()
    int2_all = r['int2_block_scores'].flatten().numpy()
    upper_all = r['upper_bound_scores'].flatten().numpy()

    bins = np.linspace(min(fp_all.min(), int2_all.min()), max(upper_all.max(), fp_all.max()), 50)
    ax4.hist(fp_all, bins=bins, alpha=0.5, label='FP Scores', color='blue', density=True)
    ax4.hist(int2_all, bins=bins, alpha=0.5, label='INT2 Scores', color='green', density=True)
    ax4.hist(upper_all, bins=bins, alpha=0.5, label='MinMax Upper Bound', color='orange', density=True)
    ax4.axvline(x=r['fp_threshold'][0].mean().item(), color='k', linestyle='--', label='Avg Threshold')
    ax4.set_xlabel('Attention Score')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Layer {r["layer_idx"]} Score Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'filtering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path / 'filtering_comparison.png'}")


def visualize_heatmaps(results_list: List[Dict], output_path: Path):
    """
    Create heatmap visualizations showing which blocks are pruned.
    """
    # Select a few representative layers
    layer_indices = [0, len(results_list)//4, len(results_list)//2, 3*len(results_list)//4, len(results_list)-1]
    selected_results = [r for r in results_list if r['layer_idx'] in [results_list[i]['layer_idx'] for i in layer_indices if i < len(results_list)]]

    if not selected_results:
        selected_results = results_list[:5]

    fig, axes = plt.subplots(len(selected_results), 4, figsize=(16, 3*len(selected_results)))

    if len(selected_results) == 1:
        axes = axes.reshape(1, -1)

    method_names = ['FP (GT)', 'INT2', 'MinMax', 'Norm']

    for i, r in enumerate(selected_results):
        layer_idx = r['layer_idx']

        # Average masks across batch dimension
        masks = [
            r['fp_mask'][0].float().numpy(),          # [HQ, num_blocks]
            r['int2_mask'][0].float().numpy(),
            r['coarse_mask_minmax'][0].float().numpy(),
            r['coarse_mask_norm'][0].float().numpy(),
        ]

        for j, (mask, name) in enumerate(zip(masks, method_names)):
            ax = axes[i, j]
            # Subsample blocks for visualization
            max_blocks_show = 200
            if mask.shape[1] > max_blocks_show:
                step = mask.shape[1] // max_blocks_show
                mask = mask[:, ::step]

            im = ax.imshow(mask, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f'L{layer_idx} {name}\n(Keep: {mask.mean()*100:.1f}%)', fontsize=10)
            if j == 0:
                ax.set_ylabel('Head')
            if i == len(selected_results) - 1:
                ax.set_xlabel('Block')

    plt.tight_layout()
    plt.savefig(output_path / 'filtering_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmaps to {output_path / 'filtering_heatmaps.png'}")


def visualize_score_difference(results_list: List[Dict], output_path: Path):
    """
    Visualize the gap between upper bound and actual scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gap distribution across layers
    ax1 = axes[0, 0]
    minmax_gaps = []
    norm_gaps = []
    layers = []

    for r in results_list:
        fp = r['fp_block_scores'].flatten()
        minmax = r['upper_bound_scores'].flatten()
        norm = r['norm_based_scores'].flatten()

        minmax_gaps.append((minmax - fp).mean().item())
        norm_gaps.append((norm - fp).mean().item())
        layers.append(r['layer_idx'])

    ax1.plot(layers, minmax_gaps, 'o-', label='MinMax - FP Gap', color='orange')
    ax1.plot(layers, norm_gaps, 's-', label='Norm - FP Gap', color='red')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Score Gap')
    ax1.set_title('Upper Bound Tightness by Layer')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. INT2 quantization error
    ax2 = axes[0, 1]
    int2_errors = []
    for r in results_list:
        fp = r['fp_block_scores'].flatten()
        int2 = r['int2_block_scores'].flatten()
        int2_errors.append((int2 - fp).abs().mean().item())

    ax2.bar(layers, int2_errors, color='green', alpha=0.7)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('INT2 Quantization Error by Layer')
    ax2.grid(axis='y', alpha=0.3)

    # 3. False positive/negative analysis for middle layer
    mid_idx = len(results_list) // 2
    r = results_list[mid_idx]

    ax3 = axes[1, 0]
    fp_prune = ~r['fp_mask'][0]  # True = should prune
    int2_prune = ~r['int2_mask'][0]
    minmax_prune = ~r['coarse_mask_minmax'][0]

    # False negatives: FP says prune, but method keeps
    int2_fn = (fp_prune & ~int2_prune).float().mean(dim=0).numpy()
    minmax_fn = (fp_prune & ~minmax_prune).float().mean(dim=0).numpy()

    # False positives: FP says keep, but method prunes (dangerous!)
    int2_fp = (~fp_prune & int2_prune).float().mean(dim=0).numpy()
    minmax_fp = (~fp_prune & minmax_prune).float().mean(dim=0).numpy()

    blocks = np.arange(len(int2_fn))
    ax3.fill_between(blocks, int2_fp, alpha=0.3, color='red', label='INT2 False Positive')
    ax3.fill_between(blocks, -int2_fn, alpha=0.3, color='blue', label='INT2 False Negative')
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.set_xlabel('Block Index')
    ax3.set_ylabel('Error Rate (per head avg)')
    ax3.set_title(f'Layer {r["layer_idx"]} INT2 Error Analysis')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    ax4.fill_between(blocks, minmax_fp, alpha=0.3, color='red', label='MinMax False Positive')
    ax4.fill_between(blocks, -minmax_fn, alpha=0.3, color='blue', label='MinMax False Negative')
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.set_xlabel('Block Index')
    ax4.set_ylabel('Error Rate (per head avg)')
    ax4.set_title(f'Layer {r["layer_idx"]} MinMax Error Analysis')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'score_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved score analysis to {output_path / 'score_analysis.png'}")


def create_summary_table(results_list: List[Dict], output_path: Path):
    """
    Create a summary table of results.
    """
    summary = []
    for r in results_list:
        summary.append({
            'Layer': r['layer_idx'],
            'FP Prune %': f"{r['fp_prune_ratio']*100:.1f}",
            'INT2 Prune %': f"{r['int2_prune_ratio']*100:.1f}",
            'MinMax Prune %': f"{r['coarse_minmax_prune_ratio']*100:.1f}",
            'Norm Prune %': f"{r['coarse_norm_prune_ratio']*100:.1f}",
        })

    # Write to text file
    with open(output_path / 'summary.txt', 'w') as f:
        f.write("Filtering Method Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Layer':>6} {'FP%':>8} {'INT2%':>8} {'MinMax%':>10} {'Norm%':>8}\n")
        f.write("-" * 50 + "\n")
        for s in summary:
            f.write(f"{s['Layer']:>6} {s['FP Prune %']:>8} {s['INT2 Prune %']:>8} {s['MinMax Prune %']:>10} {s['Norm Prune %']:>8}\n")

        # Compute averages
        avg_fp = np.mean([r['fp_prune_ratio']*100 for r in results_list])
        avg_int2 = np.mean([r['int2_prune_ratio']*100 for r in results_list])
        avg_minmax = np.mean([r['coarse_minmax_prune_ratio']*100 for r in results_list])
        avg_norm = np.mean([r['coarse_norm_prune_ratio']*100 for r in results_list])

        f.write("-" * 50 + "\n")
        f.write(f"{'Avg':>6} {avg_fp:>8.1f} {avg_int2:>8.1f} {avg_minmax:>10.1f} {avg_norm:>8.1f}\n")

        f.write("\n\nNotes:\n")
        f.write("- FP: Full Precision (ground truth)\n")
        f.write("- INT2: 2-bit symmetric quantization filtering\n")
        f.write("- MinMax: Coarse filtering using per-dimension min/max bounds\n")
        f.write("- Norm: Coarse filtering using ||q||*||k||_max upper bound\n")
        f.write(f"- Block size: {results_list[0]['block_size']}, Delta: {results_list[0]['delta']}\n")

    print(f"Saved summary to {output_path / 'summary.txt'}")


def main():
    print("=" * 60)
    print("Block Filtering Method Comparison Analysis")
    print("=" * 60)

    # Check available layers
    available_layers = sorted([int(p.name.split('_')[1]) for p in DATA_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(available_layers)} layers: {available_layers[:5]}...{available_layers[-5:]}")

    # Analyze all layers
    results_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for layer_idx in tqdm(available_layers, desc="Analyzing layers"):
        try:
            result = analyze_layer(
                layer_idx=layer_idx,
                query_pos=-1,  # Last position
                block_size=128,
                delta=5.0,
                device=device,
            )
            results_list.append(result)
        except Exception as e:
            print(f"Error analyzing layer {layer_idx}: {e}")
            continue

    if not results_list:
        print("No results to visualize!")
        return

    print(f"\nAnalyzed {len(results_list)} layers successfully.")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_comparison(results_list, OUTPUT_DIR)
    visualize_heatmaps(results_list, OUTPUT_DIR)
    visualize_score_difference(results_list, OUTPUT_DIR)
    create_summary_table(results_list, OUTPUT_DIR)

    # Save raw results
    torch.save(results_list, OUTPUT_DIR / 'results.pt')
    print(f"\nSaved raw results to {OUTPUT_DIR / 'results.pt'}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

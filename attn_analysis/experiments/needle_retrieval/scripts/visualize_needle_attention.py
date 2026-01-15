#!/usr/bin/env python3
"""
可视化 NeedleBench 注意力分数分布，直观展示"大海捞针"场景
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

# 配置
DATA_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/layer_data")
RAW_TEXT_PATH = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/result/Llama-3.1-8B/needlebench_Length32000Depth42_origin_en_32k_0/raw_text.txt")
TOKENIZER_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
OUTPUT_DIR = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/experiments/needle_retrieval/outputs/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NEEDLE_TEXT = "Hidden on Emerald Island is the legendary Magic Essence."


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


def compute_attention_scores(layer_idx: int, block_size: int, device: str = 'cuda'):
    """计算指定层的注意力分数"""
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

    # 计算每个 block 的最大注意力分数
    fp_scores = torch.zeros(HQ, num_blocks, device=device)

    for kv_h in range(HKV):
        q_heads = q[kv_h * G : (kv_h + 1) * G]
        k_block = K_blocks[kv_h]
        scores = torch.einsum('gk,nbk->gnb', q_heads, k_block) * scale
        fp_scores[kv_h * G : (kv_h + 1) * G] = scores.max(dim=-1).values

    return fp_scores.cpu().numpy()


def main():
    print("=" * 60)
    print("可视化 NeedleBench 注意力分数 - 大海捞针")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取文本并找到 needle 位置
    with open(RAW_TEXT_PATH, 'r') as f:
        raw_text = f.read()

    needle_start, needle_end, total_tokens = find_needle_token_positions(
        TOKENIZER_PATH, raw_text, NEEDLE_TEXT
    )
    print(f"Needle token 位置: [{needle_start}, {needle_end})")
    print(f"总 token 数: {total_tokens}")

    # 分析参数
    block_sizes = [64, 128, 256]
    layers_to_plot = [0, 7, 14, 30]  # 选择几个代表性的层
    delta = 5.0

    # ========== 图1: 单层多head的注意力分数分布 ==========
    layer_idx = 14
    block_size = 64

    print(f"\n分析 Layer {layer_idx}, Block Size {block_size}...")
    scores = compute_attention_scores(layer_idx, block_size, device)
    HQ, num_blocks = scores.shape

    needle_block_start = needle_start // block_size
    needle_block_end = (needle_end - 1) // block_size
    needle_blocks = list(range(needle_block_start, needle_block_end + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 选择4个有代表性的 head
    heads_to_plot = [0, 8, 16, 23]

    for idx, head in enumerate(heads_to_plot):
        ax = axes[idx]
        head_scores = scores[head]

        # 计算阈值
        threshold = max(head_scores[0], head_scores[-1]) - delta

        # 画注意力分数
        x = np.arange(num_blocks)
        ax.plot(x, head_scores, 'b-', linewidth=0.8, alpha=0.7, label='Attention Score')

        # 画阈值线
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.1f})')

        # 高亮 needle 所在的 block
        for nb in needle_blocks:
            ax.axvline(x=nb, color='green', linestyle='-', linewidth=2, alpha=0.7)
            ax.scatter([nb], [head_scores[nb]], color='green', s=100, zorder=5,
                      label=f'Needle Block (score={head_scores[nb]:.1f})')

        # 标记被保留的 block (分数 >= threshold)
        kept_mask = head_scores >= threshold
        kept_blocks = np.where(kept_mask)[0]
        ax.scatter(kept_blocks, head_scores[kept_blocks], color='blue', s=20, alpha=0.5, marker='o')

        # 标注 needle 是否被保留
        needle_kept = all(head_scores[nb] >= threshold for nb in needle_blocks)
        status = "KEPT" if needle_kept else "PRUNED"
        status_color = "green" if needle_kept else "red"

        ax.set_xlabel('Block Index', fontsize=11)
        ax.set_ylabel('Max Attention Score', fontsize=11)
        ax.set_title(f'Head {head} - Needle {status}', fontsize=12, color=status_color)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 添加 needle 位置标注
        ax.annotate(f'Needle\n({needle_block_start})',
                   xy=(needle_block_start, head_scores[needle_block_start]),
                   xytext=(needle_block_start + 20, head_scores[needle_block_start] + 2),
                   fontsize=9, color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    fig.suptitle(f'Attention Scores Distribution (Layer {layer_idx}, Block Size {block_size})\n'
                f'Needle at token [{needle_start}, {needle_end}) → Block {needle_blocks}',
                fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_attention_scores_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_attention_scores_heads.png'}")

    # ========== 图2: 所有 head 平均的注意力分数 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, layer_idx in enumerate(layers_to_plot):
        ax = axes[idx]
        scores = compute_attention_scores(layer_idx, block_size, device)

        # 计算所有 head 的平均分数
        mean_scores = scores.mean(axis=0)
        std_scores = scores.std(axis=0)

        # 计算平均阈值
        thresholds = np.array([max(scores[h, 0], scores[h, -1]) - delta for h in range(scores.shape[0])])
        mean_threshold = thresholds.mean()

        x = np.arange(len(mean_scores))

        # 画分数曲线和标准差区域
        ax.fill_between(x, mean_scores - std_scores, mean_scores + std_scores,
                       alpha=0.2, color='blue', label='±1 std')
        ax.plot(x, mean_scores, 'b-', linewidth=1, label='Mean Score')

        # 画阈值
        ax.axhline(y=mean_threshold, color='r', linestyle='--', linewidth=2,
                  label=f'Mean Threshold ({mean_threshold:.1f})')

        # 高亮 needle block
        for nb in needle_blocks:
            ax.axvline(x=nb, color='green', linestyle='-', linewidth=2, alpha=0.7)
        ax.scatter(needle_blocks, mean_scores[needle_blocks], color='green', s=150,
                  zorder=5, marker='*', label=f'Needle (score={mean_scores[needle_blocks[0]]:.1f})')

        # 统计保留率
        keep_count = sum(1 for h in range(scores.shape[0])
                        if all(scores[h, nb] >= thresholds[h] for nb in needle_blocks))
        retention_rate = keep_count / scores.shape[0] * 100

        ax.set_xlabel('Block Index', fontsize=11)
        ax.set_ylabel('Attention Score', fontsize=11)
        ax.set_title(f'Layer {layer_idx} - Needle Retention: {retention_rate:.0f}%', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Mean Attention Scores Across Heads (Block Size {block_size})\n'
                f'Green line = Needle position (Block {needle_blocks})', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_attention_scores_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_attention_scores_layers.png'}")

    # ========== 图3: 不同 Block Size 下的 Needle 可视化 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_idx = 14

    for idx, bs in enumerate(block_sizes):
        ax = axes[idx]
        scores = compute_attention_scores(layer_idx, bs, device)
        mean_scores = scores.mean(axis=0)

        nb_start = needle_start // bs
        nb_end = (needle_end - 1) // bs
        needle_bs = list(range(nb_start, nb_end + 1))

        thresholds = np.array([max(scores[h, 0], scores[h, -1]) - delta for h in range(scores.shape[0])])
        mean_threshold = thresholds.mean()

        x = np.arange(len(mean_scores))

        ax.plot(x, mean_scores, 'b-', linewidth=1, label='Mean Score')
        ax.axhline(y=mean_threshold, color='r', linestyle='--', linewidth=2,
                  label=f'Threshold')

        for nb in needle_bs:
            ax.axvline(x=nb, color='green', linestyle='-', linewidth=2, alpha=0.5)
        ax.scatter(needle_bs, [mean_scores[nb] for nb in needle_bs],
                  color='green', s=150, zorder=5, marker='*', label='Needle')

        keep_count = sum(1 for h in range(scores.shape[0])
                        if all(scores[h, nb] >= thresholds[h] for nb in needle_bs))
        retention_rate = keep_count / scores.shape[0] * 100

        ax.set_xlabel('Block Index', fontsize=11)
        ax.set_ylabel('Attention Score', fontsize=11)
        ax.set_title(f'Block Size = {bs}\nNeedle Block: {needle_bs}, Retention: {retention_rate:.0f}%',
                    fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Layer {layer_idx}: Needle Detection at Different Block Sizes', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_blocksize_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_blocksize_comparison.png'}")

    # ========== 图4: 详细的单 head 大海捞针可视化 ==========
    fig, ax = plt.subplots(figsize=(20, 6))

    layer_idx = 14
    block_size = 64
    head = 13  # 选一个有代表性的 head

    scores = compute_attention_scores(layer_idx, block_size, device)
    head_scores = scores[head]
    threshold = max(head_scores[0], head_scores[-1]) - delta

    x = np.arange(len(head_scores))

    # 用颜色区分保留和剪枝的 block
    colors = ['blue' if s >= threshold else 'lightgray' for s in head_scores]

    # 画柱状图
    bars = ax.bar(x, head_scores, color=colors, alpha=0.7, width=1.0, edgecolor='none')

    # 高亮 needle block
    needle_block = needle_start // block_size
    bars[needle_block].set_color('green')
    bars[needle_block].set_alpha(1.0)

    # 画阈值线
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5, label=f'Threshold = {threshold:.1f}')

    # 标注
    needle_score = head_scores[needle_block]
    ax.annotate(f'NEEDLE\nBlock {needle_block}\nScore: {needle_score:.1f}',
               xy=(needle_block, needle_score),
               xytext=(needle_block + 30, needle_score + 3),
               fontsize=11, color='green', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # 标注第一个和最后一个 block
    ax.annotate(f'First\n{head_scores[0]:.1f}', xy=(0, head_scores[0]),
               xytext=(10, head_scores[0] + 2), fontsize=9,
               arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    ax.annotate(f'Last\n{head_scores[-1]:.1f}', xy=(len(head_scores)-1, head_scores[-1]),
               xytext=(len(head_scores)-30, head_scores[-1] + 2), fontsize=9,
               arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    # 添加统计信息
    kept_count = sum(1 for s in head_scores if s >= threshold)
    pruned_count = len(head_scores) - kept_count
    needle_status = "KEPT" if needle_score >= threshold else "PRUNED"

    textstr = f'Total Blocks: {len(head_scores)}\n' \
              f'Kept (blue): {kept_count} ({kept_count/len(head_scores)*100:.1f}%)\n' \
              f'Pruned (gray): {pruned_count} ({pruned_count/len(head_scores)*100:.1f}%)\n' \
              f'Needle Status: {needle_status}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)

    ax.set_xlabel('Block Index', fontsize=12)
    ax.set_ylabel('Max Attention Score in Block', fontsize=12)
    ax.set_title(f'Needle in a Haystack Visualization\n'
                f'Layer {layer_idx}, Head {head}, Block Size {block_size}', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-1, len(head_scores))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_haystack_detail.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_haystack_detail.png'}")

    # ========== 图5: 折线图展示所有 head 的 needle 分数 vs 阈值 ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_idx = 14
    block_size = 64
    scores = compute_attention_scores(layer_idx, block_size, device)
    HQ = scores.shape[0]

    needle_block = needle_start // block_size
    needle_scores_per_head = scores[:, needle_block]
    thresholds_per_head = np.array([max(scores[h, 0], scores[h, -1]) - delta for h in range(HQ)])

    x = np.arange(HQ)

    ax.plot(x, needle_scores_per_head, 'go-', linewidth=2, markersize=8, label='Needle Score')
    ax.plot(x, thresholds_per_head, 'r--', linewidth=2, label='Threshold')

    # 填充保留/剪枝区域
    for h in range(HQ):
        if needle_scores_per_head[h] >= thresholds_per_head[h]:
            ax.scatter([h], [needle_scores_per_head[h]], color='green', s=100, zorder=5)
        else:
            ax.scatter([h], [needle_scores_per_head[h]], color='red', s=100, zorder=5, marker='x')

    # 统计
    kept = sum(1 for h in range(HQ) if needle_scores_per_head[h] >= thresholds_per_head[h])

    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Attention Score', fontsize=12)
    ax.set_title(f'Needle Score vs Threshold Across All Heads\n'
                f'Layer {layer_idx}, Block Size {block_size} | '
                f'Needle Kept: {kept}/{HQ} heads ({kept/HQ*100:.0f}%)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "needle_vs_threshold_per_head.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR / 'needle_vs_threshold_per_head.png'}")

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

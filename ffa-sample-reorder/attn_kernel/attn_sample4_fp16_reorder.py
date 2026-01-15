# Sample4 + FP16 + Reorder 加速方案
# - 将每个 Block128 的 4 个采样点交换到序列维度的最前面
# - 无需额外存储采样 K，直接从重排后的 K 读取
# - 采样点布局: k[:, 0:num_blocks*4, ...] 存储所有采样点
#   - block i 的采样点在位置 [i*4 : (i+1)*4]
# - 剩余 token 在采样点之后
#
# 数据布局（reorder 后）:
# - k_reordered: [B, T, HKV, K]
#   - k[:, 0:num_blocks*4, ...] = 采样点 (按 block 顺序)
#   - k[:, num_blocks*4:, ...] = 剩余 token
# - v_reordered: [B, T, HKV, V] 同样重排
# - index_map: [T] 记录重排后位置到原始位置的映射（用于调试）

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# 采样点在 Block 内的偏移（均匀采样 4 个点）
# Block Size = 128, 采样间隔 = 32, 位置 = [0, 32, 64, 96]
SAMPLE_OFFSETS = [0, 32, 64, 96]
NUM_SAMPLES = 4


def reorder_kv_for_sampling(
    k: torch.Tensor,
    v: torch.Tensor,
    BS: int = 128,
    sample_offsets: list = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 K 和 V 重排，把采样点交换到序列最前面。

    Args:
        k: [B, T, HKV, K] 原始 K cache
        v: [B, T, HKV, V] 原始 V cache
        BS: Block Size，默认 128
        sample_offsets: 采样点在 block 内的偏移，默认 [0, 32, 64, 96]

    Returns:
        k_reordered: [B, T, HKV, K] 重排后的 K
        v_reordered: [B, T, HKV, V] 重排后的 V
        reorder_indices: [T] 重排索引 (reordered[i] = original[reorder_indices[i]])
        inverse_indices: [T] 逆重排索引 (original[i] 在 reordered 中的位置)
    """
    if sample_offsets is None:
        sample_offsets = SAMPLE_OFFSETS

    B, T, HKV, K_dim = k.shape
    _, _, _, V_dim = v.shape
    num_blocks = (T + BS - 1) // BS
    num_samples = len(sample_offsets)
    total_samples = num_blocks * num_samples

    device = k.device

    # 构建重排索引
    # 前 total_samples 个位置存放采样点
    # 后面存放剩余 token
    reorder_indices = torch.zeros(T, dtype=torch.long, device=device)
    inverse_indices = torch.zeros(T, dtype=torch.long, device=device)

    # 标记哪些位置是采样点
    sample_mask = torch.zeros(T, dtype=torch.bool, device=device)
    sample_positions = []

    for block_idx in range(num_blocks):
        block_start = block_idx * BS
        for sample_idx, offset in enumerate(sample_offsets):
            pos = block_start + offset
            if pos < T:
                sample_positions.append(pos)
                sample_mask[pos] = True

    # 采样点放到最前面
    sample_positions = torch.tensor(sample_positions, dtype=torch.long, device=device)
    num_actual_samples = len(sample_positions)

    # 非采样点的位置
    non_sample_positions = torch.arange(T, device=device)[~sample_mask]

    # 构建重排索引: [采样点位置..., 非采样点位置...]
    reorder_indices[:num_actual_samples] = sample_positions
    reorder_indices[num_actual_samples:] = non_sample_positions

    # 构建逆索引
    inverse_indices[reorder_indices] = torch.arange(T, device=device)

    # 重排 K 和 V
    k_reordered = k[:, reorder_indices, :, :]
    v_reordered = v[:, reorder_indices, :, :]

    return k_reordered, v_reordered, reorder_indices, inverse_indices


def get_sample_range(T: int, BS: int = 128, num_samples: int = 4) -> int:
    """
    获取采样点在重排后序列中占据的范围 [0, sample_range)。
    """
    num_blocks = (T + BS - 1) // BS
    # 最后一个 block 可能不完整，需要检查实际采样点数
    last_block_start = (num_blocks - 1) * BS
    actual_samples_last_block = sum(1 for off in SAMPLE_OFFSETS if last_block_start + off < T)
    total_samples = (num_blocks - 1) * num_samples + actual_samples_last_block
    return total_samples


@triton.jit
def attn_compute_threshold_reorder(
    q, k_reordered,
    th_out,
    scale, T, NTB, delta,
    num_samples_total,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    BK: tl.constexpr = 64,
    NUM_SAMPLES: tl.constexpr = 4,
):
    """
    计算阈值：使用第一个和最后一个 block 的采样点（从重排后的 K 读取）。

    k_reordered: [B, T, HKV, K] - 重排后的 K，采样点在前 num_samples_total 个位置
    采样点布局: block i 的采样点在位置 [i*4 : (i+1)*4]
    """
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # 第一个 block (tb0 = 0) 和最后一个 block (tb1 = NTB - 1)
    tb0 = 0
    tb1 = NTB - 1

    # 采样点在重排后 K 中的位置
    # block i 的采样点: [i*NUM_SAMPLES : (i+1)*NUM_SAMPLES]
    sample0_start = tb0 * NUM_SAMPLES  # = 0
    sample1_start = tb1 * NUM_SAMPLES

    # 初始化累加器
    b_s0 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)
    b_s1 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

    offs_k_base = tl.arange(0, BK)
    sample_offs = tl.arange(0, NUM_SAMPLES)

    # 遍历 K 维度
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K

        # 加载 Q: [BM_DOT, BK]
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        # 加载 block 0 的采样 K: [BK, NUM_SAMPLES]
        # k_reordered: [B, T, HKV, K], 采样点在位置 [sample0_start : sample0_start+NUM_SAMPLES]
        sample0_t_offs = sample0_start + sample_offs  # [NUM_SAMPLES]
        k_base0 = k_reordered + pid_b * (T * HKV * K) + sample0_t_offs[None, :] * (HKV * K) + pid_hkv * K
        k_ptrs0 = k_base0 + offs_k[:, None]
        k_tile0 = tl.load(k_ptrs0, mask=k_mask[:, None], other=0.0).to(tl.float16)

        # 加载 block NTB-1 的采样 K: [BK, NUM_SAMPLES]
        sample1_t_offs = sample1_start + sample_offs
        sample1_valid = sample1_t_offs < num_samples_total
        k_base1 = k_reordered + pid_b * (T * HKV * K) + sample1_t_offs[None, :] * (HKV * K) + pid_hkv * K
        k_ptrs1 = k_base1 + offs_k[:, None]
        k_tile1 = tl.load(k_ptrs1, mask=k_mask[:, None] & sample1_valid[None, :], other=0.0).to(tl.float16)

        # 点积: [BM_DOT, BK] x [BK, NUM_SAMPLES] -> [BM_DOT, NUM_SAMPLES]
        b_s0 += tl.dot(q_sub, k_tile0, out_dtype=tl.float32)
        b_s1 += tl.dot(q_sub, k_tile1, out_dtype=tl.float32)

    # 应用 scale
    b_s0 = b_s0 * scale * RCP_LN2
    b_s1 = b_s1 * scale * RCP_LN2

    # 取每行的最大值
    m0 = tl.max(b_s0, axis=1)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_reorder(
    q, k_reordered, v_reordered,
    m_buf, l_buf, o_buf,
    mask_buf, stride_mask_b, stride_mask_h, stride_mask_n,
    scale, T, NTB, NTBS, delta,
    th_in,
    num_samples_total,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,
    BK: tl.constexpr = 64,
    NUM_SAMPLES: tl.constexpr = 4,
    NONSAMPLE_PER_BLOCK: tl.constexpr = 124,  # BS - NUM_SAMPLES, 需要作为编译时常量
):
    """
    Stage1: 使用重排后 K 的采样点快速筛选，非剪枝 block 使用完整 K 精确计算。

    k_reordered: [B, T, HKV, K] - 重排后的 K
      - 位置 [0, num_samples_total) 是采样点
      - 位置 [num_samples_total, T) 是剩余 token
    v_reordered: [B, T, HKV, V] - 重排后的 V (同样布局)
    """
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k_base = tl.arange(0, BK)
    sample_offs = tl.arange(0, NUM_SAMPLES)

    # 加载或计算阈值
    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        # 内联计算阈值
        tb0 = 0
        tb1 = NTB - 1
        sample0_start = tb0 * NUM_SAMPLES
        sample1_start = tb1 * NUM_SAMPLES

        b_s0 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)
        b_s1 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            sample0_t_offs = sample0_start + sample_offs
            k_base0 = k_reordered + pid_b * (T * HKV * K) + sample0_t_offs[None, :] * (HKV * K) + pid_hkv * K
            k_ptrs0 = k_base0 + offs_k[:, None]
            k_tile0 = tl.load(k_ptrs0, mask=k_mask[:, None], other=0.0).to(tl.float16)

            sample1_t_offs = sample1_start + sample_offs
            sample1_valid = sample1_t_offs < num_samples_total
            k_base1 = k_reordered + pid_b * (T * HKV * K) + sample1_t_offs[None, :] * (HKV * K) + pid_hkv * K
            k_ptrs1 = k_base1 + offs_k[:, None]
            k_tile1 = tl.load(k_ptrs1, mask=k_mask[:, None] & sample1_valid[None, :], other=0.0).to(tl.float16)

            b_s0 += tl.dot(q_sub, k_tile0, out_dtype=tl.float32)
            b_s1 += tl.dot(q_sub, k_tile1, out_dtype=tl.float32)

        b_s0 = b_s0 * scale * RCP_LN2
        b_s1 = b_s1 * scale * RCP_LN2
        m0 = tl.max(b_s0, axis=1)
        m1 = tl.max(b_s1, axis=1)
        th_rows = tl.maximum(m0, m1) - delta

    mask_base = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h

    # 当前 block 的采样点在重排后 K 中的起始位置
    sample_start = pid_tb * NUM_SAMPLES

    # 当前 block 在重排后 K 中的完整数据位置
    # 采样点: [sample_start, sample_start + NUM_SAMPLES)
    # 非采样点: 在 num_samples_total 之后的某个位置
    # 对于 block i:
    #   - 采样点: 位置 [i*4, i*4+4) 在重排后的 [0, num_samples_total)
    #   - 非采样点: block i 原本有 BS 个 token，其中 4 个是采样点，BS-4 个是非采样点
    #   - 非采样点在重排后的位置: num_samples_total + i*(BS-NUM_SAMPLES) + [0, BS-NUM_SAMPLES)

    # 遍历当前大 block 内的所有 sub-block
    for sb in tl.static_range(NSB):
        tb_sb = pid_tb * NSB + sb

        # ===== 使用采样点快速筛选 =====
        b_s_sample = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            # 加载当前 block 的采样 K (从重排后的位置)
            sample_t_offs = sample_start + sample_offs
            sample_valid = sample_t_offs < num_samples_total
            k_base_sb = k_reordered + pid_b * (T * HKV * K) + sample_t_offs[None, :] * (HKV * K) + pid_hkv * K
            k_ptrs_sb = k_base_sb + offs_k[:, None]
            k_tile_sb = tl.load(k_ptrs_sb, mask=k_mask[:, None] & sample_valid[None, :], other=0.0).to(tl.float16)

            b_s_sample += tl.dot(q_sub, k_tile_sb, out_dtype=tl.float32)

        b_s_sample = b_s_sample * scale * RCP_LN2

        # 取采样点的最大分数作为 block 近似分数
        m_rows_sample = tl.max(b_s_sample, axis=1)

        # 判断是否剪枝
        below = (m_rows_sample < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        keep_flag = tl.where(prune_blk, 0, 1).to(tl.int8)
        tl.store(mask_base + tb_sb * stride_mask_n, keep_flag)
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # ===== 非剪枝 block：使用完整 K/V 精确计算 =====
            # 需要访问当前 block 的所有 token:
            # 1. 采样点: 位置 [sample_start, sample_start + NUM_SAMPLES)
            # 2. 非采样点: 位置 [num_samples_total + pid_tb*(BS-NUM_SAMPLES) + sb*SBS_nonsample, ...]

            # 采样点部分的计算
            b_s_sample_full = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)
            for k_start in tl.static_range(0, K, BK):
                offs_k = k_start + offs_k_base
                k_mask = offs_k < K

                q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                sample_t_offs = sample_start + sample_offs
                sample_valid = sample_t_offs < num_samples_total
                k_base = k_reordered + pid_b * (T * HKV * K) + sample_t_offs[None, :] * (HKV * K) + pid_hkv * K
                k_ptrs = k_base + offs_k[:, None]
                k_tile = tl.load(k_ptrs, mask=k_mask[:, None] & sample_valid[None, :], other=0.0).to(tl.float16)

                b_s_sample_full += tl.dot(q_sub, k_tile, out_dtype=tl.float32)

            b_s_sample_full = b_s_sample_full * scale * RCP_LN2

            # 非采样点部分
            # 每个 block 有 NONSAMPLE_PER_BLOCK 个非采样点
            # block pid_tb 的非采样点在位置: num_samples_total + pid_tb*NONSAMPLE_PER_BLOCK + [0, NONSAMPLE_PER_BLOCK)
            nonsample_start = num_samples_total + pid_tb * NONSAMPLE_PER_BLOCK

            # 计算当前 sub-block 对应的非采样点范围
            # sub-block sb 对应原 block 内的位置 [sb*SBS, (sb+1)*SBS)
            # 需要映射到非采样点
            # 简化: 我们直接处理整个 block 的非采样点

            b_s_nonsample = tl.zeros([BM_DOT, NONSAMPLE_PER_BLOCK], tl.float32)
            nonsample_offs = tl.arange(0, NONSAMPLE_PER_BLOCK)
            nonsample_t_offs = nonsample_start + nonsample_offs
            nonsample_valid = nonsample_t_offs < T

            for k_start in tl.static_range(0, K, BK):
                offs_k = k_start + offs_k_base
                k_mask = offs_k < K

                q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                k_base = k_reordered + pid_b * (T * HKV * K) + nonsample_t_offs[None, :] * (HKV * K) + pid_hkv * K
                k_ptrs = k_base + offs_k[:, None]
                k_tile = tl.load(k_ptrs, mask=k_mask[:, None] & nonsample_valid[None, :], other=0.0).to(tl.float16)

                b_s_nonsample += tl.dot(q_sub, k_tile, out_dtype=tl.float32)

            b_s_nonsample = b_s_nonsample * scale * RCP_LN2

            # 合并采样点和非采样点的分数
            # 拼接 [NUM_SAMPLES] + [nonsample_per_block] -> [BS]
            # 由于 triton 不支持动态拼接，我们分别计算 softmax 然后合并

            # 采样点的 softmax
            sample_valid_mask = (sample_start + sample_offs) < num_samples_total
            b_s_sample_masked = tl.where(sample_valid_mask[None, :], b_s_sample_full, NEG_INF)
            m_sample = tl.max(b_s_sample_masked, axis=1)

            # 非采样点的 softmax
            b_s_nonsample_masked = tl.where(nonsample_valid[None, :], b_s_nonsample, NEG_INF)
            m_nonsample = tl.max(b_s_nonsample_masked, axis=1)

            # 合并 max
            m_rows = tl.maximum(m_sample, m_nonsample)

            # 计算 exp 和 sum
            p_sample = tl.where(sample_valid_mask[None, :], tl.exp2(b_s_sample_masked - m_rows[:, None]), 0.0)
            p_nonsample = tl.where(nonsample_valid[None, :], tl.exp2(b_s_nonsample_masked - m_rows[:, None]), 0.0)

            l_sample = tl.sum(p_sample, axis=1)
            l_nonsample = tl.sum(p_nonsample, axis=1)
            l_rows = l_sample + l_nonsample

            # 加载 V 并计算输出
            o_tile = tl.zeros([BM_DOT, V], tl.float32)

            # 采样点的 V
            v_sample_ptrs = v_reordered + pid_b * (T * HKV * V) + (sample_start + sample_offs)[:, None] * (HKV * V) + pid_hkv * V + v_offs[None, :]
            v_sample = tl.load(v_sample_ptrs, mask=sample_valid_mask[:, None], other=0.0).to(tl.float16)
            o_tile += tl.dot(p_sample.to(tl.float16), v_sample, out_dtype=tl.float32)

            # 非采样点的 V
            v_nonsample_ptrs = v_reordered + pid_b * (T * HKV * V) + nonsample_t_offs[:, None] * (HKV * V) + pid_hkv * V + v_offs[None, :]
            v_nonsample = tl.load(v_nonsample_ptrs, mask=nonsample_valid[:, None], other=0.0).to(tl.float16)
            o_tile += tl.dot(p_nonsample.to(tl.float16), v_nonsample, out_dtype=tl.float32)

            m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])


@triton.jit
def attn_scatter_indices_kernel(
    mask_ptr, offsets_ptr, indices_ptr,
    stride_mask_b, stride_mask_h, stride_mask_n,
    stride_off_b, stride_off_h, stride_off_n,
    stride_idx_b, stride_idx_h, stride_idx_k,
    NTBS,
    MAX_KEPT: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    pid_blk = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    block_start = pid_blk * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < NTBS

    mask_ptrs = mask_ptr + pid_b * stride_mask_b + pid_hkv * stride_mask_h + offs * stride_mask_n
    off_ptrs = offsets_ptr + pid_b * stride_off_b + pid_hkv * stride_off_h + offs * stride_off_n
    vals_mask = tl.load(mask_ptrs, mask=mask, other=0).to(tl.int32)
    vals_offs = tl.load(off_ptrs, mask=mask, other=0).to(tl.int32)

    write_pos = tl.where(vals_mask > 0, vals_offs - 1, 0)
    in_bounds = mask & (vals_mask > 0) & (write_pos < MAX_KEPT)
    out_ptrs = indices_ptr + pid_b * stride_idx_b + pid_hkv * stride_idx_h + write_pos * stride_idx_k
    tl.store(out_ptrs, offs, mask=in_bounds)


@triton.jit
def attn_forward_stage2_compact(
    m_buf, l_buf, o_buf, kept_indices, kept_counts, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
    MAX_KEPT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)

    n_kept = tl.load(kept_counts + pid_b * HKV + pid_hkv)
    keep_base = kept_indices + pid_b * (HKV * MAX_KEPT) + pid_hkv * MAX_KEPT

    for i in range(MAX_KEPT):
        mask_i = i < n_kept
        tb = tl.load(keep_base + i, mask=mask_i, other=0)
        m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb, mask=mask_i, other=neg_inf)
        l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb, mask=mask_i, other=0.0)
        o_b = tl.load(
            o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs,
            mask=mask_i,
            other=0.0,
        )
        new_m = tl.maximum(b_m, m_b)
        r_prev = tl.exp2(b_m - new_m)
        r_blk = tl.exp2(m_b - new_m)
        b_acc = b_acc * r_prev + l_b * r_blk
        b_o = b_o * r_prev + o_b * r_blk
        b_m = new_m

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def _kernel_kwargs(num_warps: int | None, num_stages: int | None) -> dict:
    kwargs = {}
    if num_warps is not None:
        if num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {num_warps}")
        kwargs["num_warps"] = int(num_warps)
    if num_stages is not None:
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        kwargs["num_stages"] = int(num_stages)
    return kwargs


def _record_kernel_time(kernel_times: dict | None, name: str, fn, device) -> None:
    if kernel_times is None:
        fn()
        return
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    kernel_times[name] = start.elapsed_time(end)


def _resolve_max_kept(max_kept: int | None, ntbs: int, ratio: float) -> int:
    if max_kept is not None:
        if max_kept <= 0:
            raise ValueError(f"max_kept must be positive, got {max_kept}")
        return int(max_kept)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"max_kept_ratio must be in (0, 1], got {ratio}")
    return max(32, min(ntbs, int(math.ceil(ntbs * ratio))))


def attn_forward_decode_reorder(
    q: torch.Tensor,              # [B, 1, HQ, K]
    k_reordered: torch.Tensor,    # [B, T, HKV, K] 重排后的 K
    v_reordered: torch.Tensor,    # [B, T, HKV, V] 重排后的 V
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    num_warps_th: int | None = None,
    num_stages_th: int | None = None,
    num_warps_s1: int | None = None,
    num_stages_s1: int | None = None,
    num_warps_s2: int | None = None,
    num_stages_s2: int | None = None,
    max_kept: int | None = None,
    max_kept_ratio: float = 0.2,
    return_kernel_timings: bool = False,
    **kwargs,
):
    """
    Sample4 + FP16 + Reorder 加速的 decode attention。

    使用重排后 K 的采样点快速筛选 block，非剪枝 block 使用完整 K 精确计算。
    采样点已交换到序列最前面，无需额外存储。
    """
    assert q.is_cuda and k_reordered.is_cuda and v_reordered.is_cuda

    B, Tq, HQ, K = q.shape
    _, T, HKV, _ = k_reordered.shape
    _, _, _, V = v_reordered.shape

    assert Tq == 1

    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # 计算采样点总数
    num_samples_total = get_sample_range(T, BS, NUM_SAMPLES)

    max_kept = _resolve_max_kept(max_kept, NTBS, max_kept_ratio)

    assert q.is_contiguous() and k_reordered.is_contiguous() and v_reordered.is_contiguous()

    kernel_times = {} if return_kernel_timings else None
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    block_mask = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int8)
    block_offsets = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int32)
    kept_indices = torch.empty((B, HKV, max_kept), device=q.device, dtype=torch.int32)
    kept_counts = torch.empty((B, HKV), device=q.device, dtype=torch.int32)

    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)
        def _launch_threshold():
            attn_compute_threshold_reorder[(B, HKV)](
                q, k_reordered,
                threshold_buf,
                scale, T, NTB, delta,
                num_samples_total,
                B=B, HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
                NUM_SAMPLES=NUM_SAMPLES,
                **th_kwargs,
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    nonsample_per_block = BS - NUM_SAMPLES
    def _launch_stage1():
        attn_forward_stage1_reorder[(NTB, B, HKV)](
            q, k_reordered, v_reordered,
            m_buf, l_buf, o_buf,
            block_mask, block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            num_samples_total,
            B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
            USE_EXT_TH=use_ext_th, NUM_SAMPLES=NUM_SAMPLES,
            NONSAMPLE_PER_BLOCK=nonsample_per_block,
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    def _launch_scan():
        torch.cumsum(block_mask, dim=-1, dtype=torch.int32, out=block_offsets)
    _record_kernel_time(kernel_times, "scan", _launch_scan, q.device)

    if NTBS > 0:
        kept_counts.copy_(block_offsets.select(-1, NTBS - 1))
    else:
        kept_counts.zero_()

    scatter_block = 256
    def _launch_scatter():
        grid = (triton.cdiv(NTBS, scatter_block), B, HKV)
        attn_scatter_indices_kernel[grid](
            block_mask, block_offsets, kept_indices,
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            block_offsets.stride(0), block_offsets.stride(1), block_offsets.stride(2),
            kept_indices.stride(0), kept_indices.stride(1), kept_indices.stride(2),
            NTBS,
            MAX_KEPT=max_kept,
            BLOCK=scatter_block,
        )
    _record_kernel_time(kernel_times, "scatter", _launch_scatter, q.device)

    skip_ratio = None
    if return_skip_ratio:
        kept = kept_counts.sum()
        total = float(kept_counts.numel() * NTBS)
        skip_ratio = float((1.0 - (kept.float() / total)).item())

    s2_kwargs = _kernel_kwargs(num_warps_s2, num_stages_s2)
    def _launch_stage2():
        attn_forward_stage2_compact[(B, HKV, G)](
            m_buf, l_buf, o_buf,
            kept_indices, kept_counts,
            o, NTBS,
            B=B, HKV=HKV, G=G, HQ=HQ, V=V,
            MAX_KEPT=max_kept,
            **s2_kwargs,
        )
    _record_kernel_time(kernel_times, "stage2", _launch_stage2, q.device)

    if return_skip_ratio:
        if return_kernel_timings:
            return o, skip_ratio, kernel_times
        return o, skip_ratio
    if return_kernel_timings:
        return o, kernel_times
    return o


class CUDAGraphDecodeRunnerReorder:
    """Capture and replay the Sample4+FP16+Reorder decode kernel with static buffers."""

    def __init__(
        self,
        q: torch.Tensor,
        k_reordered: torch.Tensor,
        v_reordered: torch.Tensor,
        *,
        precomputed_threshold: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        max_kept: int | None = None,
        max_kept_ratio: float = 0.2,
        warmup: int = 2,
        num_warps_th: Optional[int] = None,
        num_stages_th: Optional[int] = None,
        num_warps_s1: Optional[int] = None,
        num_stages_s1: Optional[int] = None,
        num_warps_s2: Optional[int] = None,
        num_stages_s2: Optional[int] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDAGraph capture.")
        if q.device.type != "cuda":
            raise ValueError("q must be a CUDA tensor.")

        self._device = q.device
        self._scale = scale
        self._BS = BS
        self._SBS = SBS
        self._delta = delta
        self._use_ext_th = precomputed_threshold is not None
        self._num_warps_th = num_warps_th
        self._num_stages_th = num_stages_th
        self._num_warps_s1 = num_warps_s1
        self._num_stages_s1 = num_stages_s1
        self._num_warps_s2 = num_warps_s2
        self._num_stages_s2 = num_stages_s2

        _, T, _, _ = k_reordered.shape
        sbs = BS if SBS is None else SBS
        ntb = triton.cdiv(T, BS)
        nsb = triton.cdiv(BS, sbs)
        ntbs = ntb * nsb
        self._max_kept = _resolve_max_kept(max_kept, ntbs, max_kept_ratio)

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_k = torch.empty_like(k_reordered, device=self._device)
        self._static_v = torch.empty_like(v_reordered, device=self._device)

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(
                precomputed_threshold, device=self._device
            )

        self._static_q.copy_(q)
        self._static_k.copy_(k_reordered)
        self._static_v.copy_(v_reordered)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        for _ in range(max(1, warmup)):
            attn_forward_decode_reorder(
                q=self._static_q,
                k_reordered=self._static_k,
                v_reordered=self._static_v,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
                num_warps_s2=self._num_warps_s2,
                num_stages_s2=self._num_stages_s2,
            )
        torch.cuda.synchronize(self._device)

        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            self._static_out = attn_forward_decode_reorder(
                q=self._static_q,
                k_reordered=self._static_k,
                v_reordered=self._static_v,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
                num_warps_s2=self._num_warps_s2,
                num_stages_s2=self._num_stages_s2,
            )

    @property
    def output(self) -> torch.Tensor:
        return self._static_out

    def replay(
        self,
        q: torch.Tensor,
        k_reordered: torch.Tensor,
        v_reordered: torch.Tensor,
        *,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")

        self._static_q.copy_(q)
        self._static_k.copy_(k_reordered)
        self._static_v.copy_(v_reordered)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        _, skip_ratio = attn_forward_decode_reorder(
            q=self._static_q,
            k_reordered=self._static_k,
            v_reordered=self._static_v,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            max_kept=self._max_kept,
            return_skip_ratio=True,
            precomputed_threshold=self._static_threshold,
            num_warps_th=self._num_warps_th,
            num_stages_th=self._num_stages_th,
            num_warps_s1=self._num_warps_s1,
            num_stages_s1=self._num_stages_s1,
            num_warps_s2=self._num_warps_s2,
            num_stages_s2=self._num_stages_s2,
        )
        return self._static_out, skip_ratio

    __call__ = replay

    def replay_only(self) -> torch.Tensor:
        """Replay without updating static inputs."""
        self._graph.replay()
        return self._static_out

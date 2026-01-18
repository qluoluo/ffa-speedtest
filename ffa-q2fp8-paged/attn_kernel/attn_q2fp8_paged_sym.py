# 分页量化 Attention Kernel：对称量化 + 低寄存器 BK=64 + 紧凑 keep 列表 + Paged KV Cache
# - 对称量化：仅使用 k_scale（无 zero-point），用 QZERO 抵消量化偏置。
# - K 维度按 BK=64 分块（低寄存器路径），替代完整的 K_PACKED 展开。
# - Stage1 先算阈值 th_rows，再按 (tb, sb) 计算块内每个 row 的最大值；
#   若所有 row 都低于阈值则 prune，否则写入 m/l/o，并写出 block_mask (0/1)。
# - Host 侧对 block_mask 做 cumsum 得到 kept_counts + 写入位置，再用 scatter kernel 填充 kept_indices（无原子操作）。
# - Stage2 仅遍历 kept_indices[0:n_kept] 合并输出，不需要扫描全 NTBS；列表顺序不保证，但不影响最终归约。
# - Paged KV Cache: KV 按页面存储，通过 page_table 映射逻辑位置到物理页面

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym"


@triton.jit
def attn_compute_threshold_qbits_paged(
    q, k_q, k_scale,
    page_table,  # [B, max_pages_per_seq]
    seq_lens,    # [B]
    th_out,
    scale, T_max, NTB_max, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 16,
    MAX_PAGES_PER_SEQ: tl.constexpr = 8192,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
):
    """计算阈值：基于每个 batch 的第一个 page 和最后一个 page

    每个 batch 独立计算自己的 threshold:
    - 第一个 page: page_table[b, 0]
    - 最后一个 page: page_table[b, num_pages - 1]
    - threshold = max(first_page_max, last_page_max) - delta
    """
    # 2D grid = (B, HKV)
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Get actual sequence length for this batch
    T = tl.load(seq_lens + pid_b)

    # 计算该 batch 的 page 数量
    num_pages = (T + PAGE_SIZE - 1) // PAGE_SIZE

    # ========== 第一个 page ==========
    # 获取第一个物理页面索引
    first_phys_page = tl.load(page_table + pid_b * MAX_PAGES_PER_SEQ + 0)

    # 第一个 page 内的 token 偏移
    offs_in_page0 = tl.arange(0, PAGE_SIZE)
    # 第一个 page 的 token 在整个序列中的位置是 0 ~ PAGE_SIZE-1
    # 需要 mask 掉超出序列长度的 token
    t_mask0 = offs_in_page0 < T

    # k_q 的物理地址：k_q[first_phys_page, offs_in_page, pid_hkv, :]
    base_tok0_q = first_phys_page * (PAGE_SIZE * HKV * K_PACKED) + offs_in_page0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)

    # ========== 最后一个 page ==========
    # 获取最后一个物理页面索引
    last_page_idx = tl.maximum(num_pages - 1, 0)
    last_phys_page = tl.load(page_table + pid_b * MAX_PAGES_PER_SEQ + last_page_idx)

    # 最后一个 page 内的 token 偏移
    offs_in_page1 = tl.arange(0, PAGE_SIZE)
    # 最后一个 page 的 token 在整个序列中的位置是 last_page_idx * PAGE_SIZE + offs_in_page1
    global_offs1 = last_page_idx * PAGE_SIZE + offs_in_page1
    t_mask1 = global_offs1 < T

    base_tok1_q = last_phys_page * (PAGE_SIZE * HKV * K_PACKED) + offs_in_page1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)

    # ========== 计算 scale 基地址 ==========
    if USE_PERBLOCK_SCALE:
        # per-block scale 不太适用于 page-based threshold，使用全局 scale
        scale_base = pid_b * (HKV * K) + pid_hkv * K
    else:
        scale_base = pid_b * (HKV * K) + pid_hkv * K

    # ========== 计算 Q·K 点积 ==========
    b_s0 = tl.zeros([BM_DOT, PAGE_SIZE], tl.float32)
    b_s1 = tl.zeros([BM_DOT, PAGE_SIZE], tl.float32)
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)

    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        pack_idx = offs_k // VALS_PER_BYTE
        pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

        # 加载 Q
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        # 加载 scale 并预乘
        scale_sub = tl.load(k_scale + scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)

        # 第一个 page 的 K
        kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
        kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
        kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s0 += tl.dot(q_scaled_sub, kq_tile0, out_dtype=tl.float32)

        # 最后一个 page 的 K
        kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
        kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
        kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s1 += tl.dot(q_scaled_sub, kq_tile1, out_dtype=tl.float32)

    # 应用 zero-point 校正和 scale
    q_zero_sum *= -QZERO
    b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    # 计算 threshold: max(first_page, last_page) - delta
    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_fused_threshold_qbits_compact_paged(
    q, k_q, k_scale, k_res, v,
    page_table,  # [B, max_pages_per_seq]
    seq_lens,    # [B]
    m_buf, l_buf, o_buf,
    mask_buf, stride_mask_b, stride_mask_h, stride_mask_n,
    scale, T_max, NTB_max, NTBS, delta,
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 16,
    MAX_PAGES_PER_SEQ: tl.constexpr = 8192,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
    MAX_KEPT: tl.constexpr = 256,
    BK: tl.constexpr = 64,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
):
    """Stage1: 计算每个 sub-block 的注意力分数，根据阈值剪枝

    变长序列处理：
    1. 每个 batch 使用自己的 seq_len 来确定有效的 block 范围
    2. 超出序列长度的 block 直接标记为 pruned (mask=0)
    3. 内联 threshold 计算时使用每个 batch 自己的第一个和最后一个 PAGE
    4. 每个 batch 独立使用自己的 threshold 进行剪枝
    """
    # 3D grid = (NTB_max, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    # Get actual sequence length for this batch
    T = tl.load(seq_lens + pid_b)

    # 计算这个 batch 实际的 NTB
    NTB_local = (T + BS - 1) // BS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # 检查当前 block 是否超出该 batch 的序列范围
    # 如果整个 block 都超出范围，直接将所有 sub-block 标记为 pruned
    block_out_of_range = pid_tb >= NTB_local

    mask_base = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h

    # 如果整个 block 超出范围，直接标记所有 sub-block 为 pruned 并返回
    if block_out_of_range:
        for sb in tl.static_range(NSB):
            tb_sb = pid_tb * NSB + sb
            tl.store(mask_base + tb_sb * stride_mask_n, tl.zeros((), tl.int8))
        return

    # Compute scale base pointer based on quantization mode
    if USE_PERBLOCK_SCALE:
        # k_scale: [B, NTB_max, HKV, K]
        scale_base = pid_b * (NTB_max * HKV * K) + pid_tb * (HKV * K) + pid_hkv * K
    else:
        # k_scale: [B, HKV, K]
        scale_base = pid_b * (HKV * K) + pid_hkv * K

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        th_rows = tl.zeros([BM_DOT], tl.float32)

    # Precompute q·QZERO once per (B, HKV).
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)
    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
        scale_sub = tl.load(k_scale + scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    if not USE_EXT_TH:
        # 计算 threshold：基于该 batch 的第一个 page 和最后一个 page
        # 计算该 batch 的 page 数量
        num_pages = (T + PAGE_SIZE - 1) // PAGE_SIZE

        # ========== 第一个 page ==========
        first_phys_page = tl.load(page_table + pid_b * MAX_PAGES_PER_SEQ + 0)
        offs_in_page0 = tl.arange(0, PAGE_SIZE)
        t_mask0 = offs_in_page0 < T
        base_tok0_q = first_phys_page * (PAGE_SIZE * HKV * K_PACKED) + offs_in_page0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)

        # ========== 最后一个 page ==========
        last_page_idx = tl.maximum(num_pages - 1, 0)
        last_phys_page = tl.load(page_table + pid_b * MAX_PAGES_PER_SEQ + last_page_idx)
        offs_in_page1 = tl.arange(0, PAGE_SIZE)
        global_offs1 = last_page_idx * PAGE_SIZE + offs_in_page1
        t_mask1 = global_offs1 < T
        base_tok1_q = last_phys_page * (PAGE_SIZE * HKV * K_PACKED) + offs_in_page1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)

        # 计算 Q·K 点积
        b_s0 = tl.zeros([BM_DOT, PAGE_SIZE], tl.float32)
        b_s1 = tl.zeros([BM_DOT, PAGE_SIZE], tl.float32)
        q_zero_sum_th = tl.zeros([BM_DOT], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            scale_sub = tl.load(k_scale + scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
            q_zero_sum_th += tl.sum(q_scaled_sub.to(tl.float32), axis=1)

            # 第一个 page 的 K
            kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
            kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
            kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s0 += tl.dot(q_scaled_sub, kq_tile0, out_dtype=tl.float32)

            # 最后一个 page 的 K
            kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
            kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
            kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s1 += tl.dot(q_scaled_sub, kq_tile1, out_dtype=tl.float32)

        q_zero_sum_th *= -QZERO
        b_s0 = (b_s0 + q_zero_sum_th[:, None]) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        b_s1 = (b_s1 + q_zero_sum_th[:, None]) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # Paged: map to physical pages
        page_idx_sb = offs_t_sb // PAGE_SIZE
        in_page_offset_sb = offs_t_sb % PAGE_SIZE
        page_table_ptrs_sb = page_table + pid_b * MAX_PAGES_PER_SEQ + page_idx_sb
        phys_page_sb = tl.load(page_table_ptrs_sb, mask=t_mask_sb, other=0)

        base_toksb_q = phys_page_sb * (PAGE_SIZE * HKV * K_PACKED) + in_page_offset_sb * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        base_toksb_k = phys_page_sb * (PAGE_SIZE * HKV * K) + in_page_offset_sb * (HKV * K) + (pid_hkv * K)

        b_s_q = tl.zeros([BM_DOT, SBS], tl.float32)
        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
            scale_sub = tl.load(k_scale + scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)

            kq_ptrssb = k_q + base_toksb_q[None, :] + pack_idx[:, None]
            kq_tilesb = tl.load(kq_ptrssb, mask=k_mask[:, None] & t_mask_sb[None, :], other=0).to(tl.int32)
            kq_tilesb = ((kq_tilesb >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s_q += tl.dot(q_scaled_sub, kq_tilesb, out_dtype=tl.float32)

        b_s_q = b_s_q + q_zero_sum[:, None]
        b_s_q_scaled = b_s_q * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s_q_scaled, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        keep_flag = tl.where(prune_blk, 0, 1).to(tl.int8)
        tl.store(mask_base + tb_sb * stride_mask_n, keep_flag)
        v_offs = tl.arange(0, V)

        if not prune_blk:
            if USE_FP8_RESIDUAL:
                b_s_res = tl.zeros([BM_DOT, SBS], tl.float32)
                for k_start in tl.static_range(0, K, BK):
                    offs_k = k_start + offs_k_base
                    k_mask = offs_k < K
                    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                    q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
                    k_res_ptrssb = k_res + base_toksb_k[None, :] + offs_k[:, None]
                    k_res_tile = tl.load(
                        k_res_ptrssb,
                        mask=k_mask[:, None] & t_mask_sb[None, :],
                        other=0.0,
                    ).to(tl.float16)
                    b_s_res += tl.dot(q_sub, k_res_tile, out_dtype=tl.float32)

                b_s = (b_s_q + b_s_res) * scale * RCP_LN2
                b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
                m_rows = tl.max(b_s, axis=1)
            else:
                b_s = b_s_q_scaled
                m_rows = m_rows_blk

            b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                # Paged: v layout is [num_pages, page_size, HKV, V]
                base_toksb_v = phys_page_sb * (PAGE_SIZE * HKV * V) + in_page_offset_sb * (HKV * V) + (pid_hkv * V)
                v_ptrs = v + base_toksb_v[:, None] + v_offs[None, :]
                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

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
    """将保留的块索引写入紧凑的 kept_indices 列表"""
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
    """Stage2: 合并保留块的输出"""
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


def _normalize_scale(k_scale: torch.Tensor, expect_shape, allow_perblock: bool = False, NTB: int = None):
    """
    Ensure scale tensors are contiguous and have expected shape.

    Args:
        k_scale: Scale tensor, either [B, HKV, K] (global) or [B, NTB, HKV, K] (per-block)
        expect_shape: Expected shape for global scale [B, HKV, K]
        allow_perblock: If True, allow per-block scale with shape [B, NTB, HKV, K]
        NTB: Number of token blocks (required if allow_perblock=True)

    Returns:
        Tuple of (normalized_scale, use_perblock_scale)
    """
    if k_scale.ndim == 4:
        if k_scale.shape[1] == 1:
            # [B, 1, HKV, K] -> [B, HKV, K]
            k_scale = k_scale.squeeze(1)
        elif allow_perblock and NTB is not None and k_scale.shape[1] == NTB:
            # Per-block scale: [B, NTB, HKV, K]
            B, _, HKV, K = k_scale.shape
            expected_perblock = (B, NTB, HKV, K)
            if k_scale.shape != expected_perblock:
                raise ValueError(
                    f"Per-block k_scale shape mismatch: {k_scale.shape=}, expected {expected_perblock}"
                )
            return k_scale.contiguous(), True

    if k_scale.shape != expect_shape:
        raise ValueError(
            f"Unsupported k_scale shape: {k_scale.shape=}, expected {expect_shape}"
        )

    return k_scale.contiguous(), False



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


def attn_forward_decode_quantized_paged(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [num_pages, page_size, HKV, K_packed], paged quantized keys
    k_scale: torch.Tensor,     # [B, HKV, K] (token dimension removed)
    v: torch.Tensor,           # [num_pages, page_size, HKV, V], paged values
    page_table: torch.Tensor,  # [B, max_pages_per_seq], maps logical page idx to physical
    seq_lens: torch.Tensor,    # [B], actual sequence length for each batch
    k_residual: torch.Tensor | None = None,  # [num_pages, page_size, HKV, K], fp8 residual (paged)
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
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
    Paged Quantized Attention Forward Pass (Decode Stage)

    Args:
        q: Query tensor [B, 1, HQ, K]
        k_q: Paged quantized key tensor [num_pages, page_size, HKV, K_packed]
        k_scale: Scale tensor for dequantization [B, HKV, K]
        v: Paged value tensor [num_pages, page_size, HKV, V]
        page_table: Page table mapping [B, max_pages_per_seq]
        seq_lens: Sequence lengths [B]
        k_residual: Optional paged FP8 residual [num_pages, page_size, HKV, K]
        k_bits: Number of bits for quantization (default 2)
        scale: Attention scale factor (default 1/sqrt(K))
        BS: Block size for token blocks
        SBS: Sub-block size (default same as BS)
        delta: Threshold delta for pruning
        return_skip_ratio: Whether to return skip ratio
        precomputed_threshold: Optional precomputed threshold tensor
        use_fp8_residual: Whether to use FP8 residual
        max_kept: Maximum number of kept blocks
        max_kept_ratio: Ratio for computing max_kept
        return_kernel_timings: Whether to return kernel timings

    Returns:
        Output tensor [B, HQ, V], optionally with skip_ratio and/or kernel_timings
    """
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 2:
        raise ValueError(f"attn_forward_decode_quantized_paged currently supports 2-bit keys, got k_bits={k_bits}")
    assert k_scale.is_cuda, "k_scale must be a CUDA tensor"
    if not k_scale.is_floating_point():
        raise ValueError("k_scale must be floating point tensor for dequantization")
    if k_q.is_floating_point():
        raise ValueError("k_q must contain integer quantized values (e.g., uint8/int8)")
    if k_residual is not None and not k_residual.is_floating_point():
        raise ValueError("k_residual must be a floating point tensor (e.g., fp8/fp16/bf16)")

    B, Tq, HQ, K = q.shape
    num_pages, page_size, HKV, K_packed = k_q.shape
    num_pages_v, page_size_v, HKVv, V = v.shape

    assert num_pages == num_pages_v and page_size == page_size_v and HKV == HKVv, \
        "k_q and v must have same paged layout"

    if 8 % k_bits != 0:
        raise ValueError(f"k_bits must divide 8 for packing, got {k_bits}")
    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    if K_packed != expected_k_packed:
        raise ValueError(f"k_q packed dim mismatch: got {K_packed}, expected {expected_k_packed} for K={K}, k_bits={k_bits}")

    if k_residual is not None:
        num_pages_r, page_size_r, HKV_r, K_r = k_residual.shape
        assert (
            num_pages == num_pages_r
            and page_size == page_size_r
            and HKV == HKV_r
            and K == K_r
        ), "k_residual must have same paged layout as k_q"

    assert Tq == 1, "Only single token decoding supported"
    G = HQ // HKV

    # page_table and seq_lens validation
    assert page_table.is_cuda and seq_lens.is_cuda
    assert page_table.shape[0] == B and seq_lens.shape[0] == B
    max_pages_per_seq = page_table.shape[1]

    # Compute max sequence length from seq_lens
    T_max = int(seq_lens.max().item())
    NTB = triton.cdiv(T_max, BS)

    expect_shape = (B, HKV, K)
    k_scale, use_perblock_scale = _normalize_scale(k_scale, expect_shape, allow_perblock=True, NTB=NTB)

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    max_kept = _resolve_max_kept(max_kept, NTBS, max_kept_ratio)

    assert q.is_contiguous() and k_q.is_contiguous() and v.is_contiguous()
    if use_fp8_residual and k_residual is None:
        raise ValueError("use_fp8_residual=True requires k_residual")
    if k_residual is not None:
        assert k_residual.is_contiguous()

    q = q.contiguous()
    k_q = k_q.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_q
    v = v.contiguous()
    page_table = page_table.contiguous()
    seq_lens = seq_lens.contiguous()

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
            attn_compute_threshold_qbits_paged[(B, HKV)](
                q, k_q, k_scale,
                page_table, seq_lens,
                threshold_buf,
                scale, T_max, NTB, delta,  # NTB 这里实际是 NTB_max
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G,
                PAGE_SIZE=page_size,
                MAX_PAGES_PER_SEQ=max_pages_per_seq,
                K_BITS=k_bits,
                USE_PERBLOCK_SCALE=use_perblock_scale,
                **th_kwargs,
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    def _launch_stage1():
        attn_forward_stage1_fused_threshold_qbits_compact_paged[(NTB, B, HKV)](
            q, k_q, k_scale, k_res, v,
            page_table, seq_lens,
            m_buf, l_buf, o_buf,
            block_mask, block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            scale, T_max, NTB, NTBS, delta,  # NTB 这里实际是 NTB_max
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            PAGE_SIZE=page_size,
            MAX_PAGES_PER_SEQ=max_pages_per_seq,
            K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual, MAX_KEPT=max_kept,
            USE_PERBLOCK_SCALE=use_perblock_scale,
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
        # 计算 skip ratio：需要考虑变长序列
        # 对于每个 batch，有效的 block 数量是 ceil(seq_len / SBS)
        # 超出范围的 block 已被 kernel 标记为 pruned (mask=0)
        kept = kept_counts.sum()
        # 计算每个 batch 实际有效的 sub-block 数量
        # ntbs_per_batch[b] = ceil(seq_lens[b] / SBS)
        ntbs_per_batch = (seq_lens.float() / SBS).ceil().int()
        total_valid_blocks = (ntbs_per_batch.sum() * HKV).float()
        if total_valid_blocks > 0:
            skip_ratio = float((1.0 - (kept.float() / total_valid_blocks)).item())
        else:
            skip_ratio = 0.0

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


class CUDAGraphDecodeRunnerQ2FP8Paged:
    """Capture and replay the Q2FP8 Paged decode kernel with static buffers.

    This wrapper avoids per-step kernel launches by using torch.cuda.CUDAGraph.
    Output is written into a persistent tensor; callers should not assume it
    survives across replays.
    """

    def __init__(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        max_kept: int | None = None,
        max_kept_ratio: float = 0.2,
        use_fp8_residual: bool = True,
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
        self._k_bits = k_bits
        self._scale = scale
        self._BS = BS
        self._SBS = SBS
        self._delta = delta
        self._use_fp8_residual = use_fp8_residual
        self._use_ext_th = precomputed_threshold is not None
        self._num_warps_th = num_warps_th
        self._num_stages_th = num_stages_th
        self._num_warps_s1 = num_warps_s1
        self._num_stages_s1 = num_stages_s1
        self._num_warps_s2 = num_warps_s2
        self._num_stages_s2 = num_stages_s2

        if self._use_fp8_residual and k_residual is None:
            raise ValueError("use_fp8_residual=True requires k_residual")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required when use_ext_th=True")

        T_max = int(seq_lens.max().item())
        sbs = BS if SBS is None else SBS
        ntb = triton.cdiv(T_max, BS)
        nsb = triton.cdiv(BS, sbs)
        ntbs = ntb * nsb
        self._max_kept = _resolve_max_kept(max_kept, ntbs, max_kept_ratio)

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_k_q = torch.empty_like(k_q, device=self._device)
        self._static_k_scale = torch.empty_like(k_scale, device=self._device)
        self._static_v = torch.empty_like(v, device=self._device)
        self._static_page_table = torch.empty_like(page_table, device=self._device)
        self._static_seq_lens = torch.empty_like(seq_lens, device=self._device)
        self._static_k_residual = None
        if self._use_fp8_residual:
            self._static_k_residual = torch.empty_like(k_residual, device=self._device)

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(
                precomputed_threshold, device=self._device
            )

        # Seed static buffers once to avoid uninitialized data in capture.
        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        self._static_page_table.copy_(page_table)
        self._static_seq_lens.copy_(seq_lens)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Warmup to trigger Triton JIT before graph capture.
        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized_paged(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                page_table=self._static_page_table,
                seq_lens=self._static_seq_lens,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
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
            self._static_out = attn_forward_decode_quantized_paged(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                page_table=self._static_page_table,
                seq_lens=self._static_seq_lens,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
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
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")
        if self._use_fp8_residual and k_residual is None:
            raise ValueError("k_residual is required for this captured graph.")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")

        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        self._static_page_table.copy_(page_table)
        self._static_seq_lens.copy_(seq_lens)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        # NOTE: Skip ratio computation is not captured; it re-runs the kernel once.
        _, skip_ratio = attn_forward_decode_quantized_paged(
            q=self._static_q,
            k_q=self._static_k_q,
            k_scale=self._static_k_scale,
            k_residual=self._static_k_residual,
            v=self._static_v,
            page_table=self._static_page_table,
            seq_lens=self._static_seq_lens,
            k_bits=self._k_bits,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            max_kept=self._max_kept,
            return_skip_ratio=True,
            precomputed_threshold=self._static_threshold,
            use_fp8_residual=self._use_fp8_residual,
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

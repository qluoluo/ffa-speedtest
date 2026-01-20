# 速度优先：对称量化 + 低寄存器 BK=64 + 紧凑 keep 列表。
# - 对称量化：仅使用 k_scale（无 zero-point），用 QZERO 抵消量化偏置。
# - K 维度按 BK=64 分块（低寄存器路径），替代完整的 K_PACKED 展开。
# - Stage1 先算阈值 th_rows，再按 (tb, sb) 计算块内每个 row 的最大值；
#   若所有 row 都低于阈值则 prune，否则写入 m/l/o，并写出 block_mask (0/1)。
# - Host 侧对 block_mask 做 cumsum 得到 kept_counts + 写入位置，再用 scatter kernel 填充 kept_indices（无原子操作）。
# - Stage2 仅遍历 kept_indices[0:n_kept] 合并输出，不需要扫描全 NTBS；列表顺序不保证，但不影响最终归约。
# CUDAGraph wrapper for Q2FP8 decode kernel (sym + compact + low-reg BK=64).
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym"

@triton.jit
def attn_compute_threshold_qbits(
    q, k_q, k_scale,
    k_current,  # NEW: [B, MAX_CURRENT, HKV, K] FP16 current tokens
    th_out,
    scale, T, NTB, delta, current_len,  # NEW: current_len
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr = 128,  # Block size for correct offset calculation
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
    MAX_CURRENT: tl.constexpr = 128,  # NEW: max current buffer size
):
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

    # Compute scale base pointer based on quantization mode
    tb0 = 0
    if USE_PERBLOCK_SCALE:
        # k_scale: [B, NTB, HKV, K]
        scale_base0 = pid_b * (NTB * HKV * K) + tb0 * (HKV * K) + pid_hkv * K
    else:
        # k_scale: [B, HKV, K]
        scale_base0 = pid_b * (HKV * K) + pid_hkv * K

    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok0_q, K_PACKED)

    tb1 = NTB - 1
    if USE_PERBLOCK_SCALE:
        scale_base1 = pid_b * (NTB * HKV * K) + tb1 * (HKV * K) + pid_hkv * K
    else:
        scale_base1 = pid_b * (HKV * K) + pid_hkv * K

    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok1_q, K_PACKED)

    b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
    b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)
    q_zero_sum0 = tl.zeros([BM_DOT], tl.float32)
    q_zero_sum1 = tl.zeros([BM_DOT], tl.float32)

    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        pack_idx = offs_k // VALS_PER_BYTE
        pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        # Load scale for first block (tb0)
        scale_sub0 = tl.load(k_scale + scale_base0 + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub0 = q_sub * scale_sub0[None, :].to(tl.float16)
        q_zero_sum0 += tl.sum(q_scaled_sub0.to(tl.float32), axis=1)

        kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
        kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
        kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s0 += tl.dot(q_scaled_sub0, kq_tile0, out_dtype=tl.float32)

        # Load scale for last block (tb1)
        scale_sub1 = tl.load(k_scale + scale_base1 + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub1 = q_sub * scale_sub1[None, :].to(tl.float16)
        q_zero_sum1 += tl.sum(q_scaled_sub1.to(tl.float32), axis=1)

        kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
        kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
        kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s1 += tl.dot(q_scaled_sub1, kq_tile1, out_dtype=tl.float32)

    q_zero_sum0 *= -QZERO
    q_zero_sum1 *= -QZERO
    b_s0 = (b_s0 + q_zero_sum0[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    b_s1 = (b_s1 + q_zero_sum1[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    # Compute max from quantized blocks
    max_quantized = tl.maximum(m0, m1)

    # NEW: Process FP16 current tokens
    max_current = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)
    if current_len > 0:
        # Process each token in current buffer
        for t in range(MAX_CURRENT):
            if t < current_len:
                score_t = tl.zeros([BM_DOT], dtype=tl.float32)

                for k_start in tl.static_range(0, K, BK):
                    offs_k = k_start + offs_k_base
                    k_mask = offs_k < K

                    # Load query
                    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                    q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                    # Load k_current[t]: [B, MAX_CURRENT, HKV, K]
                    k_curr_ptrs = k_current + pid_b * (MAX_CURRENT * HKV * K) + t * (HKV * K) + pid_hkv * K + offs_k
                    k_curr_sub = tl.load(k_curr_ptrs, mask=k_mask, other=0.0).to(tl.float16)

                    # Accumulate dot product
                    score_t += tl.sum(q_sub * k_curr_sub[None, :], axis=1)

                # Convert to log2 scale
                score_t = score_t * scale * RCP_LN2
                max_current = tl.maximum(max_current, score_t)

    # Global max across quantized and current
    global_max = tl.maximum(max_quantized, max_current)
    th_rows = global_max - delta

    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_fused_threshold_qbits_compact(
    q, k_q, k_scale, k_res, v,
    k_current, v_current,  # NEW: FP16 current tokens
    m_buf, l_buf, o_buf,
    mask_buf, stride_mask_b, stride_mask_h, stride_mask_n,
    scale, T, NTB, NTBS, delta, current_len,  # NEW: current_len
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
    MAX_KEPT: tl.constexpr = 256,
    BK: tl.constexpr = 64,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
    MAX_CURRENT: tl.constexpr = 128,  # NEW
):
    # 3D grid = (NTB + 1, B, HKV)  # +1 for FP16 current block
    # When pid_tb == NTB, process FP16 current tokens
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    # Check if this is the FP16 current block
    is_current_block = (pid_tb == NTB)

    if is_current_block:
        # Process FP16 current block
        if current_len == 0:
            # No current tokens, skip
            return

        RCP_LN2 = 1.4426950408889634
        NEG_INF = float("-inf")

        base_hq = pid_hkv * G
        rows = tl.arange(0, BM_DOT)
        row_mask = rows < G

        # Load threshold
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=NEG_INF)

        # Initialize accumulators
        m_rows = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)
        l_rows = tl.zeros([BM_DOT], dtype=tl.float32)
        o_tile = tl.zeros([BM_DOT, V], dtype=tl.float32)

        max_score_block = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)

        offs_k_base = tl.arange(0, 64)  # BK=64
        v_offs = tl.arange(0, V)

        # Process each token in current buffer
        for t in range(MAX_CURRENT):
            if t < current_len:
                # Compute score for this token
                score_t = tl.zeros([BM_DOT], dtype=tl.float32)

                for k_start in tl.static_range(0, K, 64):
                    offs_k = k_start + offs_k_base
                    k_mask = offs_k < K

                    # Load query
                    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                    q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                    # Load k_current[t]
                    k_curr_ptrs = k_current + pid_b * (MAX_CURRENT * HKV * K) + t * (HKV * K) + pid_hkv * K + offs_k
                    k_curr_sub = tl.load(k_curr_ptrs, mask=k_mask, other=0.0).to(tl.float16)

                    score_t += tl.sum(q_sub * k_curr_sub[None, :], axis=1)

                score_t = score_t * scale * RCP_LN2
                max_score_block = tl.maximum(max_score_block, score_t)

                # Online softmax update
                m_new = tl.maximum(m_rows, score_t)
                alpha = tl.exp2(m_rows - m_new)
                l_rows = l_rows * alpha + tl.exp2(score_t - m_new)

                # Load v_current[t] and update output
                v_curr_ptrs = v_current + pid_b * (MAX_CURRENT * HKV * V) + t * (HKV * V) + pid_hkv * V + v_offs
                v_curr_sub = tl.load(v_curr_ptrs, mask=(v_offs < V), other=0.0).to(tl.float16)

                o_tile = o_tile * alpha[:, None] + tl.exp2(score_t - m_new)[:, None] * v_curr_sub[None, :]
                m_rows = m_new

        # Check if block should be kept
        should_keep = tl.max(max_score_block) >= tl.max(th_rows)

        mask_base = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h
        keep_flag = tl.where(should_keep, 1, 0).to(tl.int8)
        tl.store(mask_base + NTBS * stride_mask_n, keep_flag)

        if should_keep:
            # Write outputs
            m_ptrs = m_buf + pid_b * (HQ * (NTBS + 1)) + (base_hq + rows) * (NTBS + 1) + NTBS
            l_ptrs = l_buf + pid_b * (HQ * (NTBS + 1)) + (base_hq + rows) * (NTBS + 1) + NTBS
            o_ptrs = o_buf + pid_b * (HQ * (NTBS + 1) * V) + (base_hq + rows)[:, None] * ((NTBS + 1) * V) + NTBS * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

        return

    # Original quantized block processing (pid_tb < NTB)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Compute scale base pointer based on quantization mode
    if USE_PERBLOCK_SCALE:
        # k_scale: [B, NTB, HKV, K]
        scale_base = pid_b * (NTB * HKV * K) + pid_tb * (HKV * K) + pid_hkv * K
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
        # Compute threshold from first and last blocks
        tb0 = 0
        if USE_PERBLOCK_SCALE:
            scale_base0 = pid_b * (NTB * HKV * K) + tb0 * (HKV * K) + pid_hkv * K
        else:
            scale_base0 = pid_b * (HKV * K) + pid_hkv * K
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok0_q, K_PACKED)

        tb1 = NTB - 1
        if USE_PERBLOCK_SCALE:
            scale_base1 = pid_b * (NTB * HKV * K) + tb1 * (HKV * K) + pid_hkv * K
        else:
            scale_base1 = pid_b * (HKV * K) + pid_hkv * K
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok1_q, K_PACKED)

        b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
        b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)
        q_zero_sum0 = tl.zeros([BM_DOT], tl.float32)
        q_zero_sum1 = tl.zeros([BM_DOT], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            scale_sub0 = tl.load(k_scale + scale_base0 + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub0 = q_sub * scale_sub0[None, :].to(tl.float16)
            q_zero_sum0 += tl.sum(q_scaled_sub0.to(tl.float32), axis=1)

            kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
            kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
            kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s0 += tl.dot(q_scaled_sub0, kq_tile0, out_dtype=tl.float32)

            scale_sub1 = tl.load(k_scale + scale_base1 + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub1 = q_sub * scale_sub1[None, :].to(tl.float16)
            q_zero_sum1 += tl.sum(q_scaled_sub1.to(tl.float32), axis=1)

            kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
            kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
            kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s1 += tl.dot(q_scaled_sub1, kq_tile1, out_dtype=tl.float32)

        q_zero_sum0 *= -QZERO
        q_zero_sum1 *= -QZERO
        b_s0 = (b_s0 + q_zero_sum0[:, None]) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        b_s1 = (b_s1 + q_zero_sum1[:, None]) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    mask_base = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb_q = pid_b * (T * HKV * K_PACKED) + offs_t_sb * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        base_toksb_k = pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + (pid_hkv * K)
        tl.multiple_of(base_toksb_q, K_PACKED)
        tl.multiple_of(base_toksb_k, K)

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
                v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
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
    HAS_CURRENT: tl.constexpr = False,  # NEW: whether we have +1 for current block
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

    # Determine actual NTBS size (with or without current block)
    actual_ntbs = NTBS + 1 if HAS_CURRENT else NTBS

    for i in range(MAX_KEPT):
        mask_i = i < n_kept
        tb = tl.load(keep_base + i, mask=mask_i, other=0)
        m_b = tl.load(m_buf + pid_b * (HQ * actual_ntbs) + pid_hq * actual_ntbs + tb, mask=mask_i, other=neg_inf)
        l_b = tl.load(l_buf + pid_b * (HQ * actual_ntbs) + pid_hq * actual_ntbs + tb, mask=mask_i, other=0.0)
        o_b = tl.load(
            o_buf + pid_b * (HQ * actual_ntbs * V) + pid_hq * (actual_ntbs * V) + tb * V + v_offs,
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


def attn_forward_decode_quantized(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [B, T, HKV, ceil(K / (8 / k_bits))], packed quantized ints
    k_scale: torch.Tensor,     # [B, HKV, K] (token dimension removed)
    v: torch.Tensor,           # [B, T, HKV, V]
    k_current: torch.Tensor | None = None,  # NEW: [B, MAX_CURRENT, HKV, K] FP16 current tokens
    v_current: torch.Tensor | None = None,  # NEW: [B, MAX_CURRENT, HKV, V] FP16 current values
    current_len: int = 0,      # NEW: actual length in current buffer
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K], fp8 residual
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
    max_current: int = 128,    # NEW: max current buffer size
    **kwargs,
):
    # import os
    # print(f"ENTER {__file__} attn_forward_decode_quantized")
    
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 2:
        raise ValueError(f"attn_forward_decode_quantized currently supports 2-bit keys, got k_bits={k_bits}")
    assert k_scale.is_cuda, "k_scale must be a CUDA tensor"
    if not k_scale.is_floating_point():
        raise ValueError("k_scale must be floating point tensor for dequantization")
    if k_q.is_floating_point():
        raise ValueError("k_q must contain integer quantized values (e.g., uint8/int8)")
    if k_residual is not None and not k_residual.is_floating_point():
        raise ValueError("k_residual must be a floating point tensor (e.g., fp8/fp16/bf16)")

    B, Tq, HQ, K = q.shape
    Bk, T, HKV, K_packed = k_q.shape
    Bv, Tv, HKVv, V = v.shape
    if 8 % k_bits != 0:
        raise ValueError(f"k_bits must divide 8 for packing, got {k_bits}")
    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    if K_packed != expected_k_packed:
        raise ValueError(f"k_q packed dim mismatch: got {K_packed}, expected {expected_k_packed} for K={K}, k_bits={k_bits}")
    if k_residual is not None:
        Bk_r, T_r, HKV_r, K_r = k_residual.shape
        assert (
            B == Bk == Bv == Bk_r
            and Tq == 1
            and Tv == T == T_r
            and HKVv == HKV == HKV_r
            and K == K_r
        ), "K/V layouts must be [B, T, HKV, D]"
    else:
        assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV, "K/V layouts must be [B, T, HKV, D]"
    G = HQ // HKV

    expect_shape = (B, HKV, K)
    k_scale, use_perblock_scale = _normalize_scale(k_scale, expect_shape, allow_perblock=True, NTB=triton.cdiv(T, BS))

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    # CRITICAL FIX: Ensure SBS is a power of 2 for Triton compatibility
    # tl.arange(0, SBS) requires SBS to be a power of 2
    if SBS & (SBS - 1) != 0:  # Check if not power of 2
        # Round up to next power of 2
        import math
        SBS = 1 << (SBS - 1).bit_length()
        # But don't exceed BS
        if SBS > BS:
            SBS = 1 << (BS - 1).bit_length()
            if SBS > BS:
                SBS = SBS // 2

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # NEW: Handle k_current and v_current
    has_current = (k_current is not None) and (v_current is not None) and (current_len > 0)
    if has_current:
        assert k_current.shape == (B, max_current, HKV, K), f"k_current shape mismatch: {k_current.shape}"
        assert v_current.shape == (B, max_current, HKV, V), f"v_current shape mismatch: {v_current.shape}"
        assert 0 < current_len <= max_current, f"current_len must be in (0, {max_current}], got {current_len}"
        k_current = k_current.contiguous()
        v_current = v_current.contiguous()
        actual_ntbs = NTBS + 1  # +1 for current block
    else:
        # Create dummy buffers if not provided
        k_current = torch.empty((B, max_current, HKV, K), device=q.device, dtype=q.dtype)
        v_current = torch.empty((B, max_current, HKV, V), device=q.device, dtype=q.dtype)
        current_len = 0
        actual_ntbs = NTBS

    max_kept = _resolve_max_kept(max_kept, actual_ntbs, max_kept_ratio)

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
    kernel_times = {} if return_kernel_timings else None
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    # Allocate buffers with actual_ntbs (includes +1 for current block if present)
    m_buf = torch.empty((B, HQ, actual_ntbs), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, actual_ntbs), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, actual_ntbs, V), device=q.device, dtype=torch.float32)
    block_mask = torch.empty((B, HKV, actual_ntbs), device=q.device, dtype=torch.int8)
    block_offsets = torch.empty((B, HKV, actual_ntbs), device=q.device, dtype=torch.int32)
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
            attn_compute_threshold_qbits[(B, HKV)](
                q, k_q, k_scale,
                k_current,  # NEW
                threshold_buf,
                scale, T, NTB, delta, current_len,  # NEW: current_len
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G, BS=BS,
                K_BITS=k_bits,
                USE_PERBLOCK_SCALE=use_perblock_scale,
                MAX_CURRENT=max_current,  # NEW
                **th_kwargs,
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    def _launch_stage1():
        # Grid: (NTB + 1, B, HKV) if has_current else (NTB, B, HKV)
        grid_size = (NTB + 1, B, HKV) if has_current else (NTB, B, HKV)
        attn_forward_stage1_fused_threshold_qbits_compact[grid_size](
            q, k_q, k_scale, k_res, v,
            k_current, v_current,  # NEW
            m_buf, l_buf, o_buf,
            block_mask, block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            scale, T, NTB, NTBS, delta, current_len,  # NEW: current_len
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual, MAX_KEPT=max_kept,
            USE_PERBLOCK_SCALE=use_perblock_scale,
            MAX_CURRENT=max_current,  # NEW
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    def _launch_scan():
        torch.cumsum(block_mask, dim=-1, dtype=torch.int32, out=block_offsets)
    _record_kernel_time(kernel_times, "scan", _launch_scan, q.device)

    if actual_ntbs > 0:
        kept_counts.copy_(block_offsets.select(-1, actual_ntbs - 1))
    else:
        kept_counts.zero_()

    scatter_block = 256
    def _launch_scatter():
        grid = (triton.cdiv(actual_ntbs, scatter_block), B, HKV)
        attn_scatter_indices_kernel[grid](
            block_mask, block_offsets, kept_indices,
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            block_offsets.stride(0), block_offsets.stride(1), block_offsets.stride(2),
            kept_indices.stride(0), kept_indices.stride(1), kept_indices.stride(2),
            actual_ntbs,  # NEW: use actual_ntbs
            MAX_KEPT=max_kept,
            BLOCK=scatter_block,
        )
    _record_kernel_time(kernel_times, "scatter", _launch_scatter, q.device)

    skip_ratio = None
    if return_skip_ratio:
        kept = kept_counts.sum()
        total = float(kept_counts.numel() * actual_ntbs)  # NEW: use actual_ntbs
        skip_ratio = float((1.0 - (kept.float() / total)).item())

    s2_kwargs = _kernel_kwargs(num_warps_s2, num_stages_s2)
    def _launch_stage2():
        attn_forward_stage2_compact[(B, HKV, G)](
            m_buf, l_buf, o_buf,
            kept_indices, kept_counts,
            o, NTBS,
            B=B, HKV=HKV, G=G, HQ=HQ, V=V,
            MAX_KEPT=max_kept,
            HAS_CURRENT=has_current,  # NEW
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

class CUDAGraphDecodeRunnerQ2FP8:
    """Capture and replay the Q2FP8 decode kernel with static buffers.

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
        *,
        k_current: Optional[torch.Tensor] = None,
        v_current: Optional[torch.Tensor] = None,
        current_len: int = 0,
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
        max_current: int = 128,
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
        self._current_len = current_len
        self._max_current = max_current
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

        _, T, _, _ = k_q.shape
        sbs = BS if SBS is None else SBS
        ntb = triton.cdiv(T, BS)
        nsb = triton.cdiv(BS, sbs)
        ntbs = ntb * nsb
        self._max_kept = _resolve_max_kept(max_kept, ntbs, max_kept_ratio)

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_k_q = torch.empty_like(k_q, device=self._device)
        self._static_k_scale = torch.empty_like(k_scale, device=self._device)
        self._static_v = torch.empty_like(v, device=self._device)
        self._static_k_residual = None
        if self._use_fp8_residual:
            self._static_k_residual = torch.empty_like(k_residual, device=self._device)

        self._static_k_current = None
        self._static_v_current = None
        if k_current is not None:
            self._static_k_current = torch.empty_like(k_current, device=self._device)
        if v_current is not None:
            self._static_v_current = torch.empty_like(v_current, device=self._device)

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
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._static_k_current is not None:
            self._static_k_current.copy_(k_current)
        if self._static_v_current is not None:
            self._static_v_current.copy_(v_current)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Warmup to trigger Triton JIT before graph capture.
        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_current=self._static_k_current,
                v_current=self._static_v_current,
                current_len=self._current_len,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                max_current=self._max_current,
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
            self._static_out = attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_current=self._static_k_current,
                v_current=self._static_v_current,
                current_len=self._current_len,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                max_current=self._max_current,
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
        *,
        k_current: Optional[torch.Tensor] = None,
        v_current: Optional[torch.Tensor] = None,
        current_len: Optional[int] = None,
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
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._static_k_current is not None and k_current is not None:
            self._static_k_current.copy_(k_current)
        if self._static_v_current is not None and v_current is not None:
            self._static_v_current.copy_(v_current)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Update current_len if provided
        if current_len is not None:
            self._current_len = current_len

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        # NOTE: Skip ratio computation is not captured; it re-runs the kernel once.
        _, skip_ratio = attn_forward_decode_quantized(
            q=self._static_q,
            k_q=self._static_k_q,
            k_scale=self._static_k_scale,
            k_residual=self._static_k_residual,
            v=self._static_v,
            k_current=self._static_k_current,
            v_current=self._static_v_current,
            current_len=self._current_len,
            k_bits=self._k_bits,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            max_kept=self._max_kept,
            max_current=self._max_current,
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

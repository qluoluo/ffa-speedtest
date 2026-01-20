# H100 优先：对称量化 + 低寄存器 BK=64 + 紧凑 keep 列表 + 额外优化。
# - 预计算 q_zero_sum / q_abs_sum（q_stats）供 stage1/threshold 复用。
# - 可选 upper-bound prune：用 q_abs_sum 与 k_block_absmax 在进入 b_s_q 前快速剪枝。
# - BM_DOT 支持 4/8/16（自动按 G 选择）减少无效 row。
# - o_buf 使用 fp16/bf16 存储，stage2 转 fp32 累加，降低带宽。
# - Stage1 先算阈值 th_rows，再按 (tb, sb) 计算块内每个 row 的最大值；
#   若所有 row 都低于阈值则 prune，否则写入 m/l/o，并写出 block_mask (0/1)。
# - Host 侧对 block_mask 做 cumsum 得到 kept_counts + 写入位置，再用 scatter kernel 填充 kept_indices（无原子操作）。
# - Stage2 仅遍历 kept_indices[0:n_kept] 合并输出，不需要扫描全 NTBS；列表顺序不保证，但不影响最终归约。
# CUDAGraph wrapper for Q2FP8 decode kernel (sym + compact + low-reg BK=64 + H100 opts).
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym"

@triton.jit
def attn_compute_q_stats(
    q, k_scale,
    q_zero_out, q_abs_out,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr,
    G: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
):
    # 2D grid = (B, HKV)
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    scale_base = k_scale + pid_b * (HKV * K) + pid_hkv * K
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)
    q_abs_sum = tl.zeros([BM_DOT], tl.float32)

    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
        scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled = q_sub * scale_sub[None, :].to(tl.float16)
        q_scaled_f = q_scaled.to(tl.float32)
        q_zero_sum += tl.sum(q_scaled_f, axis=1)
        q_abs_sum += tl.sum(tl.abs(q_scaled_f), axis=1)

    q_zero_sum *= -QZERO
    q_zero_ptrs = q_zero_out + pid_b * HQ + (base_hq + rows)
    q_abs_ptrs = q_abs_out + pid_b * HQ + (base_hq + rows)
    tl.store(q_zero_ptrs, q_zero_sum, mask=row_mask)
    tl.store(q_abs_ptrs, q_abs_sum, mask=row_mask)

@triton.jit
def attn_compute_threshold_qbits(
    q, k_q, k_scale, q_zero_in,
    th_out,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
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

    # Preload base pointers for scale since they are reused in BK blocks.
    scale_base = k_scale + pid_b * (HKV * K) + pid_hkv * K

    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok0_q, K_PACKED)

    tb1 = NTB - 1
    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok1_q, K_PACKED)

    b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
    b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)
    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        pack_idx = offs_k // VALS_PER_BYTE
        pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
        kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
        kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s0 += tl.dot(q_scaled_sub, kq_tile0, out_dtype=tl.float32)

        kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
        kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
        kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s1 += tl.dot(q_scaled_sub, kq_tile1, out_dtype=tl.float32)

    q_zero_sum = tl.load(q_zero_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_compute_k_block_absmax_qbits(
    k_q, k_block_max,
    stride_kb_b, stride_kb_h, stride_kb_n,
    T,
    B: tl.constexpr, HKV: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    BS: tl.constexpr, SBS: tl.constexpr,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    offs_k_base = tl.arange(0, BK)

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb_q = pid_b * (T * HKV * K_PACKED) + offs_t_sb * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_toksb_q, K_PACKED)

        block_max = tl.zeros((), tl.float32)
        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            kq_ptrs = k_q + base_toksb_q[None, :] + pack_idx[:, None]
            kq_tile = tl.load(kq_ptrs, mask=k_mask[:, None] & t_mask_sb[None, :], other=0).to(tl.int32)
            kq_tile = ((kq_tile >> pack_shifts[:, None]) & QMAX).to(tl.float32)
            kq_tile = tl.abs(kq_tile - QZERO)

            max_k = tl.max(kq_tile, axis=0)
            block_max = tl.maximum(block_max, tl.max(max_k, axis=0))

        tb_sb = pid_tb * NSB + sb
        out_ptr = k_block_max + pid_b * stride_kb_b + pid_hkv * stride_kb_h + tb_sb * stride_kb_n
        tl.store(out_ptr, block_max)


@triton.jit
def attn_forward_stage1_fused_threshold_qbits_compact(
    q, k_q, k_scale, q_zero_in, q_abs_in, k_res, v,
    k_block_max, stride_kb_b, stride_kb_h, stride_kb_n,
    m_buf, l_buf, o_buf,
    mask_buf, stride_mask_b, stride_mask_h, stride_mask_n,
    scale, T, NTB, NTBS, delta,
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
    USE_UPPER_BOUND: tl.constexpr = False,
    MAX_KEPT: tl.constexpr = 256,
    BK: tl.constexpr = 64,
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Preload base pointers for scale since they are reused in BK blocks.
    scale_base = k_scale + pid_b * (HKV * K) + pid_hkv * K

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        th_rows = tl.zeros([BM_DOT], tl.float32)

    q_zero_sum = tl.load(q_zero_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    q_abs_sum = tl.load(q_abs_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    n_valid = tl.sum(row_mask.to(tl.int32), axis=0)

    offs_k_base = tl.arange(0, BK)

    if not USE_EXT_TH:
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok0_q, K_PACKED)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok1_q, K_PACKED)

        b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
        b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
            scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)

            kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
            kq_tile0 = tl.load(kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0).to(tl.int32)
            kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s0 += tl.dot(q_scaled_sub, kq_tile0, out_dtype=tl.float32)

            kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
            kq_tile1 = tl.load(kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0).to(tl.int32)
            kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s1 += tl.dot(q_scaled_sub, kq_tile1, out_dtype=tl.float32)

        b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
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

        tb_sb = pid_tb * NSB + sb
        prune_blk = False
        if USE_UPPER_BOUND:
            block_max = tl.load(
                k_block_max + pid_b * stride_kb_b + pid_hkv * stride_kb_h + tb_sb * stride_kb_n,
                mask=tb_sb < NTBS,
                other=0.0,
            )
            ub_scaled = q_abs_sum * block_max * scale * RCP_LN2
            ub_below = (ub_scaled < th_rows) & row_mask
            prune_blk = tl.sum(ub_below.to(tl.int32), axis=0) == n_valid

        if prune_blk:
            tl.store(mask_base + tb_sb * stride_mask_n, tl.full((), 0, tl.int8))
        else:
            b_s_q = tl.zeros([BM_DOT, SBS], tl.float32)
            for k_start in tl.static_range(0, K, BK):
                offs_k = k_start + offs_k_base
                k_mask = offs_k < K
                pack_idx = offs_k // VALS_PER_BYTE
                pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

                q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
                scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
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
            prune_blk = n_below == n_valid

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
        ).to(tl.float32)
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


def _normalize_scale(k_scale: torch.Tensor, expect_shape):
    """
    Ensure scale tensor is contiguous and has shape [B, HKV, K].
    """
    if k_scale.ndim == 4 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)

    if k_scale.shape != expect_shape:
        raise ValueError(
            f"Unsupported k_scale shape: {k_scale.shape=}, expected {expect_shape}"
        )

    return k_scale.contiguous()



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
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K], fp8 residual
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    precomputed_q_zero_sum: torch.Tensor | None = None,
    precomputed_q_abs_sum: torch.Tensor | None = None,
    k_block_max: torch.Tensor | None = None,
    use_upper_bound: bool = True,
    bm_dot: int | None = None,
    o_buf_dtype: torch.dtype | None = None,
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
    if bm_dot is None:
        if G <= 4:
            bm_dot = 4
        elif G <= 8:
            bm_dot = 8
        else:
            bm_dot = 16
    if bm_dot not in (4, 8, 16):
        raise ValueError(f"bm_dot must be one of (4, 8, 16), got {bm_dot}")
    if bm_dot < G:
        raise ValueError(f"bm_dot must be >= G={G}, got {bm_dot}")

    expect_shape = (B, HKV, K)
    k_scale = _normalize_scale(k_scale, expect_shape)

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
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
    kernel_times = {} if return_kernel_timings else None
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    if o_buf_dtype is None:
        o_buf_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=o_buf_dtype)
    block_mask = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int8)
    block_offsets = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int32)
    kept_indices = torch.empty((B, HKV, max_kept), device=q.device, dtype=torch.int32)
    kept_counts = torch.empty((B, HKV), device=q.device, dtype=torch.int32)

    q_zero_buf = precomputed_q_zero_sum
    q_abs_buf = precomputed_q_abs_sum
    if (precomputed_q_zero_sum is None) ^ (precomputed_q_abs_sum is None):
        raise ValueError("precomputed_q_zero_sum and precomputed_q_abs_sum must both be provided.")
    if q_zero_buf is not None:
        assert q_zero_buf.is_cuda and q_zero_buf.shape == (B, HQ)
        q_zero_buf = q_zero_buf.contiguous()
    else:
        q_zero_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
    if q_abs_buf is not None:
        assert q_abs_buf.is_cuda and q_abs_buf.shape == (B, HQ)
        q_abs_buf = q_abs_buf.contiguous()
    else:
        q_abs_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)

    need_q_stats = precomputed_q_zero_sum is None or precomputed_q_abs_sum is None
    if need_q_stats:
        qs_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)
        def _launch_q_stats():
            attn_compute_q_stats[(B, HKV)](
                q, k_scale,
                q_zero_buf, q_abs_buf,
                B=B, HKV=HKV, HQ=HQ, K=K, G=G,
                BM_DOT=bm_dot, K_BITS=k_bits,
                **qs_kwargs,
            )
        _record_kernel_time(kernel_times, "q_stats", _launch_q_stats, q.device)
    elif kernel_times is not None:
        kernel_times["q_stats"] = None

    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)
        def _launch_threshold():
            attn_compute_threshold_qbits[(B, HKV)](
                q, k_q, k_scale, q_zero_buf,
                threshold_buf,
                scale, T, NTB, delta,
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G,
                BM_DOT=bm_dot, K_BITS=k_bits,
                **th_kwargs,
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    if use_upper_bound:
        if k_block_max is not None:
            assert k_block_max.is_cuda and k_block_max.shape == (B, HKV, NTBS)
            k_block_buf = k_block_max.contiguous()
            if kernel_times is not None:
                kernel_times["k_block_max"] = None
        else:
            k_block_buf = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.float32)
            kb_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
            def _launch_k_block():
                attn_compute_k_block_absmax_qbits[(NTB, B, HKV)](
                    k_q, k_block_buf,
                    k_block_buf.stride(0), k_block_buf.stride(1), k_block_buf.stride(2),
                    T,
                    B=B, HKV=HKV, K=K, K_PACKED=K_packed, BS=BS, SBS=SBS,
                    K_BITS=k_bits,
                    **kb_kwargs,
                )
            _record_kernel_time(kernel_times, "k_block_max", _launch_k_block, q.device)
    else:
        k_block_buf = block_mask
        if kernel_times is not None:
            kernel_times["k_block_max"] = None

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    def _launch_stage1():
        attn_forward_stage1_fused_threshold_qbits_compact[(NTB, B, HKV)](
            q, k_q, k_scale, q_zero_buf, q_abs_buf, k_res, v,
            k_block_buf, k_block_buf.stride(0), k_block_buf.stride(1), k_block_buf.stride(2),
            m_buf, l_buf, o_buf,
            block_mask, block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            BM_DOT=bm_dot, K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual,
            USE_UPPER_BOUND=use_upper_bound, MAX_KEPT=max_kept,
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
        use_upper_bound: bool = True,
        bm_dot: int | None = None,
        o_buf_dtype: torch.dtype | None = None,
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
        self._use_upper_bound = use_upper_bound
        self._bm_dot = bm_dot
        self._o_buf_dtype = o_buf_dtype
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
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                use_upper_bound=self._use_upper_bound,
                bm_dot=self._bm_dot,
                o_buf_dtype=self._o_buf_dtype,
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
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                use_upper_bound=self._use_upper_bound,
                bm_dot=self._bm_dot,
                o_buf_dtype=self._o_buf_dtype,
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
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

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
            k_bits=self._k_bits,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            max_kept=self._max_kept,
            return_skip_ratio=True,
            precomputed_threshold=self._static_threshold,
            use_fp8_residual=self._use_fp8_residual,
            use_upper_bound=self._use_upper_bound,
            bm_dot=self._bm_dot,
            o_buf_dtype=self._o_buf_dtype,
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

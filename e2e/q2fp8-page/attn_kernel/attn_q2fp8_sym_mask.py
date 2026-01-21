# CUDAGraph wrapper for Q2FP8 decode kernel (symmetric quantization).
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
    th_out,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr = 128,  # ADD: block size for correct offset calculation
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
):
    # 2D grid = (B, HKV)
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K = tl.full([K], True, tl.int1)
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_kp = tl.arange(0, K_PACKED)
    offs_k0 = offs_kp * VALS_PER_BYTE + 0
    offs_k1 = offs_kp * VALS_PER_BYTE + 1
    offs_k2 = offs_kp * VALS_PER_BYTE + 2
    offs_k3 = offs_kp * VALS_PER_BYTE + 3
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K

    q_ptrs0 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k0[None, :]
    q_ptrs1 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k1[None, :]
    q_ptrs2 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k2[None, :]
    q_ptrs3 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k3[None, :]
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    # Load scale for first block (tb0=0)
    tb0 = 0
    if USE_PERBLOCK_SCALE:
        # k_scale: [B, NTB, HKV, K]
        scale_base0 = pid_b * (NTB * HKV * K) + tb0 * (HKV * K) + pid_hkv * K
    else:
        # k_scale: [B, HKV, K]
        scale_base0 = pid_b * (HKV * K) + pid_hkv * K
    scale_ptrs0_0 = k_scale + scale_base0 + offs_k0
    scale_ptrs0_1 = k_scale + scale_base0 + offs_k1
    scale_ptrs0_2 = k_scale + scale_base0 + offs_k2
    scale_ptrs0_3 = k_scale + scale_base0 + offs_k3
    scale0_0 = tl.load(scale_ptrs0_0, mask=mask_k0, other=0.0).to(tl.float32)
    scale0_1 = tl.load(scale_ptrs0_1, mask=mask_k1, other=0.0).to(tl.float32)
    scale0_2 = tl.load(scale_ptrs0_2, mask=mask_k2, other=0.0).to(tl.float32)
    scale0_3 = tl.load(scale_ptrs0_3, mask=mask_k3, other=0.0).to(tl.float32)

    q_scaled0_0 = q0 * scale0_0[None, :].to(tl.float16)
    q_scaled0_1 = q1 * scale0_1[None, :].to(tl.float16)
    q_scaled0_2 = q2 * scale0_2[None, :].to(tl.float16)
    q_scaled0_3 = q3 * scale0_3[None, :].to(tl.float16)
    q_zero_sum0 = tl.sum(q_scaled0_0.to(tl.float32), axis=1)
    q_zero_sum0 += tl.sum(q_scaled0_1.to(tl.float32), axis=1)
    q_zero_sum0 += tl.sum(q_scaled0_2.to(tl.float32), axis=1)
    q_zero_sum0 += tl.sum(q_scaled0_3.to(tl.float32), axis=1)
    q_zero_sum0 *= -QZERO

    # Sample first T_BS tokens of the entire sequence [0, T_BS)
    offs_t0 = tl.arange(0, T_BS)  # [0, 1, ..., T_BS-1]
    t_mask0 = offs_t0 < T
    base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok0_q, K_PACKED)
    kq_ptrs0 = k_q + base_tok0_q[None, :] + offs_kp[:, None]
    kq_packed0 = tl.load(kq_ptrs0, mask=t_mask0[None, :], other=0).to(tl.int32)
    kq0_0 = ((kq_packed0 >> 0) & QMAX).to(tl.float16)
    kq0_1 = ((kq_packed0 >> 2) & QMAX).to(tl.float16)
    kq0_2 = ((kq_packed0 >> 4) & QMAX).to(tl.float16)
    kq0_3 = ((kq_packed0 >> 6) & QMAX).to(tl.float16)
    b_s0 = tl.dot(q_scaled0_0, kq0_0, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled0_1, kq0_1, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled0_2, kq0_2, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled0_3, kq0_3, out_dtype=tl.float32)
    b_s0 = (b_s0 + q_zero_sum0[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # Load scale for last block (tb1=NTB-1)
    tb1 = NTB - 1
    if USE_PERBLOCK_SCALE:
        scale_base1 = pid_b * (NTB * HKV * K) + tb1 * (HKV * K) + pid_hkv * K
    else:
        scale_base1 = pid_b * (HKV * K) + pid_hkv * K
    scale_ptrs1_0 = k_scale + scale_base1 + offs_k0
    scale_ptrs1_1 = k_scale + scale_base1 + offs_k1
    scale_ptrs1_2 = k_scale + scale_base1 + offs_k2
    scale_ptrs1_3 = k_scale + scale_base1 + offs_k3
    scale1_0 = tl.load(scale_ptrs1_0, mask=mask_k0, other=0.0).to(tl.float32)
    scale1_1 = tl.load(scale_ptrs1_1, mask=mask_k1, other=0.0).to(tl.float32)
    scale1_2 = tl.load(scale_ptrs1_2, mask=mask_k2, other=0.0).to(tl.float32)
    scale1_3 = tl.load(scale_ptrs1_3, mask=mask_k3, other=0.0).to(tl.float32)

    q_scaled1_0 = q0 * scale1_0[None, :].to(tl.float16)
    q_scaled1_1 = q1 * scale1_1[None, :].to(tl.float16)
    q_scaled1_2 = q2 * scale1_2[None, :].to(tl.float16)
    q_scaled1_3 = q3 * scale1_3[None, :].to(tl.float16)
    q_zero_sum1 = tl.sum(q_scaled1_0.to(tl.float32), axis=1)
    q_zero_sum1 += tl.sum(q_scaled1_1.to(tl.float32), axis=1)
    q_zero_sum1 += tl.sum(q_scaled1_2.to(tl.float32), axis=1)
    q_zero_sum1 += tl.sum(q_scaled1_3.to(tl.float32), axis=1)
    q_zero_sum1 *= -QZERO

    # Sample last T_BS tokens of the entire sequence [T-T_BS, T)
    offs_t1 = T - T_BS + tl.arange(0, T_BS)  # [T-T_BS, T-T_BS+1, ..., T-1]
    t_mask1 = offs_t1 >= 0  # Handle case when T < T_BS
    base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok1_q, K_PACKED)
    kq_ptrs1 = k_q + base_tok1_q[None, :] + offs_kp[:, None]
    kq_packed1 = tl.load(kq_ptrs1, mask=t_mask1[None, :], other=0).to(tl.int32)
    kq1_0 = ((kq_packed1 >> 0) & QMAX).to(tl.float16)
    kq1_1 = ((kq_packed1 >> 2) & QMAX).to(tl.float16)
    kq1_2 = ((kq_packed1 >> 4) & QMAX).to(tl.float16)
    kq1_3 = ((kq_packed1 >> 6) & QMAX).to(tl.float16)
    b_s1 = tl.dot(q_scaled1_0, kq1_0, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled1_1, kq1_1, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled1_2, kq1_2, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled1_3, kq1_3, out_dtype=tl.float32)
    b_s1 = (b_s1 + q_zero_sum1[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_fused_threshold_qbits(
    q, k_q, k_scale, k_res, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
    USE_PERBLOCK_SCALE: tl.constexpr = False,  # NEW: support per-block scale
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_kp = tl.arange(0, K_PACKED)
    offs_k0 = offs_kp * VALS_PER_BYTE + 0
    offs_k1 = offs_kp * VALS_PER_BYTE + 1
    offs_k2 = offs_kp * VALS_PER_BYTE + 2
    offs_k3 = offs_kp * VALS_PER_BYTE + 3
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K

    q_ptrs0 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k0[None, :]
    q_ptrs1 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k1[None, :]
    q_ptrs2 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k2[None, :]
    q_ptrs3 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k3[None, :]
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    # Load scale for current block
    if USE_PERBLOCK_SCALE:
        # k_scale: [B, NTB, HKV, K]
        scale_base = pid_b * (NTB * HKV * K) + pid_tb * (HKV * K) + pid_hkv * K
    else:
        # k_scale: [B, HKV, K]
        scale_base = pid_b * (HKV * K) + pid_hkv * K
    scale_ptrs0 = k_scale + scale_base + offs_k0
    scale_ptrs1 = k_scale + scale_base + offs_k1
    scale_ptrs2 = k_scale + scale_base + offs_k2
    scale_ptrs3 = k_scale + scale_base + offs_k3
    scale0 = tl.load(scale_ptrs0, mask=mask_k0, other=0.0).to(tl.float32)
    scale1 = tl.load(scale_ptrs1, mask=mask_k1, other=0.0).to(tl.float32)
    scale2 = tl.load(scale_ptrs2, mask=mask_k2, other=0.0).to(tl.float32)
    scale3 = tl.load(scale_ptrs3, mask=mask_k3, other=0.0).to(tl.float32)

    q_scaled0 = q0 * scale0[None, :].to(tl.float16)
    q_scaled1 = q1 * scale1[None, :].to(tl.float16)
    q_scaled2 = q2 * scale2[None, :].to(tl.float16)
    q_scaled3 = q3 * scale3[None, :].to(tl.float16)
    q_zero_sum = tl.sum(q_scaled0.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled1.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled2.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled3.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        # NOTE: This branch is currently not used (USE_EXT_TH is always True)
        # but kept for completeness. WARNING: does not support USE_PERBLOCK_SCALE.
        # Sample first T_BS tokens of the sequence
        offs_t0 = tl.arange(0, T_BS)  # [0, 1, ..., T_BS-1]
        t_mask0 = offs_t0 < T
        base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok0_q, K_PACKED)
        kq_ptrs0 = k_q + base_tok0_q[None, :] + offs_kp[:, None]
        kq_packed0 = tl.load(kq_ptrs0, mask=t_mask0[None, :], other=0).to(tl.int32)
        kq0_0 = ((kq_packed0 >> 0) & QMAX).to(tl.float16)
        kq0_1 = ((kq_packed0 >> 2) & QMAX).to(tl.float16)
        kq0_2 = ((kq_packed0 >> 4) & QMAX).to(tl.float16)
        kq0_3 = ((kq_packed0 >> 6) & QMAX).to(tl.float16)
        b_s0 = tl.dot(q_scaled0, kq0_0, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled1, kq0_1, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled2, kq0_2, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled3, kq0_3, out_dtype=tl.float32)
        b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        # Sample last T_BS tokens of the sequence
        offs_t1 = T - T_BS + tl.arange(0, T_BS)  # [T-T_BS, ..., T-1]
        t_mask1 = offs_t1 >= 0
        base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok1_q, K_PACKED)
        kq_ptrs1 = k_q + base_tok1_q[None, :] + offs_kp[:, None]
        kq_packed1 = tl.load(kq_ptrs1, mask=t_mask1[None, :], other=0).to(tl.int32)
        kq1_0 = ((kq_packed1 >> 0) & QMAX).to(tl.float16)
        kq1_1 = ((kq_packed1 >> 2) & QMAX).to(tl.float16)
        kq1_2 = ((kq_packed1 >> 4) & QMAX).to(tl.float16)
        kq1_3 = ((kq_packed1 >> 6) & QMAX).to(tl.float16)
        b_s1 = tl.dot(q_scaled0, kq1_0, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled1, kq1_1, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled2, kq1_2, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled3, kq1_3, out_dtype=tl.float32)
        b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb_q = pid_b * (T * HKV * K_PACKED) + offs_t_sb * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        base_toksb_k = pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + (pid_hkv * K)
        tl.multiple_of(base_toksb_q, K_PACKED)
        tl.multiple_of(base_toksb_k, K)
        kq_ptrssb = k_q + base_toksb_q[None, :] + offs_kp[:, None]
        kq_packedsb = tl.load(kq_ptrssb, mask=t_mask_sb[None, :], other=0).to(tl.int32)
        kqsb0 = ((kq_packedsb >> 0) & QMAX).to(tl.float16)
        kqsb1 = ((kq_packedsb >> 2) & QMAX).to(tl.float16)
        kqsb2 = ((kq_packedsb >> 4) & QMAX).to(tl.float16)
        kqsb3 = ((kq_packedsb >> 6) & QMAX).to(tl.float16)
        b_s_q = tl.dot(q_scaled0, kqsb0, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled1, kqsb1, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled2, kqsb2, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled3, kqsb3, out_dtype=tl.float32)
        b_s_q = (b_s_q + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s_q, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            if USE_FP8_RESIDUAL:
                k_res_ptrs0 = k_res + base_toksb_k[None, :] + offs_k0[:, None]
                k_res_ptrs1 = k_res + base_toksb_k[None, :] + offs_k1[:, None]
                k_res_ptrs2 = k_res + base_toksb_k[None, :] + offs_k2[:, None]
                k_res_ptrs3 = k_res + base_toksb_k[None, :] + offs_k3[:, None]
                k_res0 = tl.load(
                    k_res_ptrs0,
                    mask=(mask_k0[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res1 = tl.load(
                    k_res_ptrs1,
                    mask=(mask_k1[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res2 = tl.load(
                    k_res_ptrs2,
                    mask=(mask_k2[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res3 = tl.load(
                    k_res_ptrs3,
                    mask=(mask_k3[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                # Reuse selector b_s_q and add residual dot to avoid recomputing q·k_tile_q.
                b_s_res = tl.dot(q0, k_res0, out_dtype=tl.float32)
                b_s_res += tl.dot(q1, k_res1, out_dtype=tl.float32)
                b_s_res += tl.dot(q2, k_res2, out_dtype=tl.float32)
                b_s_res += tl.dot(q3, k_res3, out_dtype=tl.float32)
                b_s_res = b_s_res * scale * RCP_LN2
                b_s = b_s_q + b_s_res
                b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
                m_rows = tl.max(b_s, axis=1)
            else:
                b_s = b_s_q
                m_rows = m_rows_blk

            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            tl.store(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
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
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)
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


@triton.jit
def attn_forward_stage2_masked_with_lse(
    m_buf, l_buf, o_buf, mask_buf, o, final_m, final_l, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    """Stage2 kernel that also outputs final m (max) and l (sum of exp) for accurate merging."""
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)
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
    # Store final m and l for accurate merging with k_current
    m_ptr = final_m + pid_b * HQ + pid_hq
    l_ptr = final_l + pid_b * HQ + pid_hq
    tl.store(m_ptr, b_m)
    tl.store(l_ptr, b_acc)


def _normalize_scale(k_scale: torch.Tensor, expect_shape, allow_perblock: bool = False, NTB: int = None):
    """
    Ensure scale tensors are contiguous and have expected shape.

    Args:
        k_scale: Scale tensor, either [B, HKV, K] (global) or [B, NTB, HKV, K] (per-block)
        expect_shape: Expected shape for global scale [B, HKV, K]
        allow_perblock: If True, allow per-block scale [B, NTB, HKV, K]
        NTB: Number of blocks (required if allow_perblock=True)

    Returns:
        Tuple of (normalized_scale, use_perblock_scale)
    """
    if k_scale.ndim == 4:
        if k_scale.shape[1] == 1:
            # [B, 1, HKV, K] -> squeeze to [B, HKV, K]
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
    return_lse: bool = False,  # NEW: return log-sum-exp (m, l) for accurate merging
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
    num_warps_th: int | None = None,
    num_stages_th: int | None = None,
    num_warps_s1: int | None = None,
    num_stages_s1: int | None = None,
    num_warps_s2: int | None = None,
    num_stages_s2: int | None = None,
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
        raise ValueError("k_scale must be a floating point tensor for dequantization")
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
    NTB = triton.cdiv(T, BS)
    k_scale, use_perblock_scale = _normalize_scale(k_scale, expect_shape, allow_perblock=True, NTB=NTB)

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

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
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((B, HKV, NTBS), device=q.device, dtype=torch.int8)

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
                threshold_buf,
                scale, T, NTB, delta,
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G,
                BS=BS,  # ADD: pass block size for correct offset calculation
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
        attn_forward_stage1_fused_threshold_qbits[(NTB, B, HKV)](
            q, k_q, k_scale, k_res, v,
            m_buf, l_buf, o_buf,
            mask_buf,
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual,
            USE_PERBLOCK_SCALE=use_perblock_scale,
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    s2_kwargs = _kernel_kwargs(num_warps_s2, num_stages_s2)

    # Allocate final m and l buffers if needed for accurate merging
    final_m = None
    final_l = None
    if return_lse:
        final_m = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        final_l = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        def _launch_stage2():
            attn_forward_stage2_masked_with_lse[(B, HKV, G)](
                m_buf, l_buf, o_buf,
                mask_buf,
                o, final_m, final_l, NTBS,
                B=B, HKV=HKV, G=G, HQ=HQ, V=V,
                **s2_kwargs,
            )
    else:
        def _launch_stage2():
            attn_forward_stage2_masked[(B, HKV, G)](
                m_buf, l_buf, o_buf,
                mask_buf,
                o, NTBS,
                B=B, HKV=HKV, G=G, HQ=HQ, V=V,
                **s2_kwargs,
            )
    _record_kernel_time(kernel_times, "stage2", _launch_stage2, q.device)

    # Build return value based on flags
    if return_lse:
        # Return (output, m, l) or (output, m, l, skip_ratio) or with kernel_timings
        if return_skip_ratio:
            if return_kernel_timings:
                return o, final_m, final_l, skip_ratio, kernel_times
            return o, final_m, final_l, skip_ratio
        if return_kernel_timings:
            return o, final_m, final_l, kernel_times
        return o, final_m, final_l
    else:
        if return_skip_ratio:
            if return_kernel_timings:
                return o, skip_ratio, kernel_times
            return o, skip_ratio
        if return_kernel_timings:
            return o, kernel_times
        return o

class CUDAGraphDecodeRunnerQ2FP8:
    """Capture and replay the Q2FP8 decode kernel with static buffers.

    OPTIMIZATION: This wrapper uses CUDA Graph to eliminate kernel launch overhead.
    Key principle: CUDA Graph requires fixed memory addresses. We capture the graph
    using the cache's own pre-allocated buffers directly (NO COPIES during replay).

    Only Q is copied (small, 1 token), while K/V cache buffers are used directly.
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

        # OPTIMIZATION: Only allocate buffer for Q (small, 1 token)
        # K/V cache buffers are used directly from the cache (no copy needed)
        self._static_q = torch.empty_like(q, device=self._device)

        # Store references to cache buffers (these are pre-allocated and won't move)
        self._cache_k_q = k_q
        self._cache_k_scale = k_scale
        self._cache_v = v
        self._cache_k_residual = k_residual if self._use_fp8_residual else None

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(
                precomputed_threshold, device=self._device
            )

        # Allocate static LSE buffers for accurate merging
        B, _, HQ, _ = q.shape
        self._static_m = torch.empty((B, HQ), device=self._device, dtype=torch.float32)
        self._static_l = torch.empty((B, HQ), device=self._device, dtype=torch.float32)

        # Seed Q buffer once to avoid uninitialized data in capture
        self._static_q.copy_(q)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Warmup to trigger Triton JIT before graph capture.
        # Use cache buffers directly (no copy needed for K/V)
        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._cache_k_q,
                k_scale=self._cache_k_scale,
                k_residual=self._cache_k_residual,
                v=self._cache_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                return_skip_ratio=False,
                return_lse=True,  # Warmup with LSE
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
            # Capture with return_lse=True to get (o, m, l)
            # Use cache buffers directly - addresses are fixed
            result = attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._cache_k_q,
                k_scale=self._cache_k_scale,
                k_residual=self._cache_k_residual,
                v=self._cache_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                return_skip_ratio=False,
                return_lse=True,  # Always capture with LSE
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
                num_warps_s2=self._num_warps_s2,
                num_stages_s2=self._num_stages_s2,
            )
            # Unpack result: (o, m, l)
            self._static_out = result[0]
            self._static_m.copy_(result[1])
            self._static_l.copy_(result[2])

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
        return_lse: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")
        if self._use_fp8_residual and k_residual is None:
            raise ValueError("k_residual is required for this captured graph.")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")

        # OPTIMIZATION: Only copy Q (small, 1 token)
        # K/V cache buffers are used directly - NO COPY needed (addresses are fixed)
        # This eliminates O(N) copy overhead where N = sequence length
        self._static_q.copy_(q)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # NOTE: Assertions removed - cache buffers are now pre-allocated with fixed addresses
        # The cache uses in-place writes, so addresses never change

        self._graph.replay()

        # Build return value based on flags
        if return_lse:
            if return_skip_ratio:
                # NOTE: Skip ratio computation is not captured; it re-runs the kernel once.
                _, _, _, skip_ratio = attn_forward_decode_quantized(
                    q=self._static_q,
                    k_q=self._cache_k_q,
                    k_scale=self._cache_k_scale,
                    k_residual=self._cache_k_residual,
                    v=self._cache_v,
                    k_bits=self._k_bits,
                    scale=self._scale,
                    BS=self._BS,
                    SBS=self._SBS,
                    delta=self._delta,
                    return_skip_ratio=True,
                    return_lse=True,
                    precomputed_threshold=self._static_threshold,
                    use_fp8_residual=self._use_fp8_residual,
                    num_warps_th=self._num_warps_th,
                    num_stages_th=self._num_stages_th,
                    num_warps_s1=self._num_warps_s1,
                    num_stages_s1=self._num_stages_s1,
                    num_warps_s2=self._num_warps_s2,
                    num_stages_s2=self._num_stages_s2,
                )
                return self._static_out, self._static_m, self._static_l, skip_ratio
            else:
                # Return without cloning to avoid overhead
                return self._static_out, self._static_m, self._static_l
        else:
            if return_skip_ratio:
                # NOTE: Skip ratio computation is not captured; it re-runs the kernel once.
                _, skip_ratio = attn_forward_decode_quantized(
                    q=self._static_q,
                    k_q=self._cache_k_q,
                    k_scale=self._cache_k_scale,
                    k_residual=self._cache_k_residual,
                    v=self._cache_v,
                    k_bits=self._k_bits,
                    scale=self._scale,
                    BS=self._BS,
                    SBS=self._SBS,
                    delta=self._delta,
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
            else:
                return self._static_out

    __call__ = replay

    def replay_only(self) -> torch.Tensor:
        """Replay without updating static inputs."""
        self._graph.replay()
        return self._static_out

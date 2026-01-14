from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .attn_q2fp8_paged import PagedKVCache

QUANT_MODE = "sym"


@triton.jit
def paged_attn_compute_threshold_qbits(
    q,
    k_q,
    k_scale,
    block_table,
    page_counts,
    page_lens,
    th_out,
    scale,
    delta,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_p,
    stride_k_h,
    stride_k_t,
    stride_k_kp,
    stride_scale_b,
    stride_scale_h,
    stride_scale_k,
    stride_bt_b,
    stride_bt_p,
    stride_pc_b,
    stride_plen_p,
    stride_th_b,
    stride_th_h,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    G: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
):
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
    offs_kp = tl.arange(0, K_PACKED)
    offs_k0 = offs_kp * VALS_PER_BYTE + 0
    offs_k1 = offs_kp * VALS_PER_BYTE + 1
    offs_k2 = offs_kp * VALS_PER_BYTE + 2
    offs_k3 = offs_kp * VALS_PER_BYTE + 3
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K

    q_ptrs0 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k0[None, :] * stride_q_k
    q_ptrs1 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k1[None, :] * stride_q_k
    q_ptrs2 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k2[None, :] * stride_q_k
    q_ptrs3 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k3[None, :] * stride_q_k
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k0 * stride_scale_k
    scale_ptrs1 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k1 * stride_scale_k
    scale_ptrs2 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k2 * stride_scale_k
    scale_ptrs3 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k3 * stride_scale_k
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

    page_count = tl.load(page_counts + pid_b * stride_pc_b)
    page_valid = page_count > 0
    page_id0 = tl.load(block_table + pid_b * stride_bt_b, mask=page_valid, other=-1).to(tl.int32)
    page_last = page_count - 1
    page_id1 = tl.load(
        block_table + pid_b * stride_bt_b + page_last * stride_bt_p,
        mask=page_valid,
        other=-1,
    ).to(tl.int32)

    offs_t = tl.arange(0, T_BS)

    page_valid0 = page_valid & (page_id0 >= 0)
    page_len0 = tl.load(page_lens + page_id0 * stride_plen_p, mask=page_valid0, other=0)
    t_mask0 = page_valid0 & (offs_t < page_len0)
    base_tok0 = page_id0 * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok0, K_PACKED)
    kq_ptrs0 = k_q + base_tok0[None, :] + offs_kp[:, None] * stride_k_kp
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

    page_valid1 = page_valid & (page_id1 >= 0)
    page_len1 = tl.load(page_lens + page_id1 * stride_plen_p, mask=page_valid1, other=0)
    t_mask1 = page_valid1 & (offs_t < page_len1)
    base_tok1 = page_id1 * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok1, K_PACKED)
    kq_ptrs1 = k_q + base_tok1[None, :] + offs_kp[:, None] * stride_k_kp
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
    th_ptrs = th_out + pid_b * stride_th_b + (base_hq + rows) * stride_th_h
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def paged_attn_compute_threshold_qbits_contig(
    q,
    k_q,
    k_scale,
    page_counts,
    page_lens,
    th_out,
    scale,
    delta,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_p,
    stride_k_h,
    stride_k_t,
    stride_k_kp,
    stride_scale_b,
    stride_scale_h,
    stride_scale_k,
    stride_pc_b,
    stride_plen_p,
    stride_th_b,
    stride_th_h,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    G: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
):
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
    offs_kp = tl.arange(0, K_PACKED)
    offs_k0 = offs_kp * VALS_PER_BYTE + 0
    offs_k1 = offs_kp * VALS_PER_BYTE + 1
    offs_k2 = offs_kp * VALS_PER_BYTE + 2
    offs_k3 = offs_kp * VALS_PER_BYTE + 3
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K

    q_ptrs0 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k0[None, :] * stride_q_k
    q_ptrs1 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k1[None, :] * stride_q_k
    q_ptrs2 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k2[None, :] * stride_q_k
    q_ptrs3 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k3[None, :] * stride_q_k
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k0 * stride_scale_k
    scale_ptrs1 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k1 * stride_scale_k
    scale_ptrs2 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k2 * stride_scale_k
    scale_ptrs3 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k3 * stride_scale_k
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

    page_count = tl.load(page_counts + pid_b * stride_pc_b)
    page_valid = page_count > 0
    page_id0 = tl.full((), 0, tl.int32)
    page_id1 = page_count - 1

    offs_t = tl.arange(0, T_BS)

    page_valid0 = page_valid
    page_len0 = tl.load(page_lens + page_id0 * stride_plen_p, mask=page_valid0, other=0)
    t_mask0 = page_valid0 & (offs_t < page_len0)
    base_tok0 = page_id0 * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok0, K_PACKED)
    kq_ptrs0 = k_q + base_tok0[None, :] + offs_kp[:, None] * stride_k_kp
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

    page_valid1 = page_valid & (page_id1 >= 0)
    page_len1 = tl.load(page_lens + page_id1 * stride_plen_p, mask=page_valid1, other=0)
    t_mask1 = page_valid1 & (offs_t < page_len1)
    base_tok1 = page_id1 * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok1, K_PACKED)
    kq_ptrs1 = k_q + base_tok1[None, :] + offs_kp[:, None] * stride_k_kp
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
    th_ptrs = th_out + pid_b * stride_th_b + (base_hq + rows) * stride_th_h
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def paged_attn_stage1_qbits(
    q,
    k_q,
    k_scale,
    v,
    block_table,
    page_counts,
    page_lens,
    th_in,
    m_buf,
    l_buf,
    o_buf,
    mask_buf,
    scale,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_p,
    stride_k_h,
    stride_k_t,
    stride_k_kp,
    stride_scale_b,
    stride_scale_h,
    stride_scale_k,
    stride_v_p,
    stride_v_h,
    stride_v_t,
    stride_v_d,
    stride_bt_b,
    stride_bt_p,
    stride_pc_b,
    stride_plen_p,
    stride_th_b,
    stride_th_h,
    stride_m_b,
    stride_m_h,
    stride_m_t,
    stride_l_b,
    stride_l_h,
    stride_l_t,
    stride_o_b,
    stride_o_h,
    stride_o_t,
    stride_o_d,
    stride_mask_b,
    stride_mask_h,
    stride_mask_t,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
):
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    page_count = tl.load(page_counts + pid_b * stride_pc_b)
    page_valid = pid_tb < page_count
    page_id = tl.load(
        block_table + pid_b * stride_bt_b + pid_tb * stride_bt_p,
        mask=page_valid,
        other=-1,
    ).to(tl.int32)
    page_valid = page_valid & (page_id >= 0)

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

    q_ptrs0 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k0[None, :] * stride_q_k
    q_ptrs1 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k1[None, :] * stride_q_k
    q_ptrs2 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k2[None, :] * stride_q_k
    q_ptrs3 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k3[None, :] * stride_q_k
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k0 * stride_scale_k
    scale_ptrs1 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k1 * stride_scale_k
    scale_ptrs2 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k2 * stride_scale_k
    scale_ptrs3 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k3 * stride_scale_k
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

    th_rows = tl.load(th_in + pid_b * stride_th_b + (base_hq + rows) * stride_th_h, mask=row_mask, other=NEG_INF)

    offs_t = tl.arange(0, SBS)
    page_len = tl.load(page_lens + page_id * stride_plen_p, mask=page_valid, other=0)
    t_mask = page_valid & (offs_t < page_len)
    base_tok = page_id * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok, K_PACKED)
    kq_ptrs = k_q + base_tok[None, :] + offs_kp[:, None] * stride_k_kp
    kq_packed = tl.load(kq_ptrs, mask=t_mask[None, :], other=0).to(tl.int32)
    kq0 = ((kq_packed >> 0) & QMAX).to(tl.float16)
    kq1 = ((kq_packed >> 2) & QMAX).to(tl.float16)
    kq2 = ((kq_packed >> 4) & QMAX).to(tl.float16)
    kq3 = ((kq_packed >> 6) & QMAX).to(tl.float16)
    b_s_q = tl.dot(q_scaled0, kq0, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled1, kq1, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled2, kq2, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled3, kq3, out_dtype=tl.float32)
    b_s_q = (b_s_q + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s_act = tl.where(t_mask[None, :], b_s_q, NEG_INF)

    m_rows_blk = tl.max(b_s_act, axis=1)
    below = (m_rows_blk < th_rows) & row_mask
    n_below = tl.sum(below.to(tl.int32), axis=0)
    n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
    prune_blk = (n_below == n_valid) | (~page_valid)

    v_offs = tl.arange(0, V)
    if not prune_blk:
        b_p = tl.where(t_mask[None, :], tl.exp2(b_s_act - m_rows_blk[:, None]), 0.0)
        l_rows = tl.sum(b_p, axis=1)

        need_v = tl.sum(t_mask.to(tl.int32), axis=0) > 0
        o_tile = tl.zeros([BM_DOT, V], tl.float32)
        if need_v:
            v_ptrs = v + page_id * stride_v_p + pid_hkv * stride_v_h + offs_t[:, None] * stride_v_t + v_offs[None, :] * stride_v_d
            b_v = tl.load(v_ptrs, mask=t_mask[:, None] & (v_offs[None, :] < V), other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

        m_ptrs = m_buf + pid_b * stride_m_b + (base_hq + rows) * stride_m_h + pid_tb * stride_m_t
        l_ptrs = l_buf + pid_b * stride_l_b + (base_hq + rows) * stride_l_h + pid_tb * stride_l_t
        o_ptrs = (
            o_buf
            + pid_b * stride_o_b
            + (base_hq + rows)[:, None] * stride_o_h
            + pid_tb * stride_o_t
            + v_offs[None, :] * stride_o_d
        )
        tl.store(m_ptrs, m_rows_blk, mask=row_mask)
        tl.store(l_ptrs, l_rows, mask=row_mask)
        tl.store(o_ptrs, o_tile, mask=row_mask[:, None] & (v_offs[None, :] < V))

    mask_ptr = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h + pid_tb * stride_mask_t
    tl.store(mask_ptr, tl.where(prune_blk, tl.full((), 0, tl.int8), tl.full((), 1, tl.int8)))


@triton.jit
def paged_attn_stage1_qbits_contig(
    q,
    k_q,
    k_scale,
    v,
    page_counts,
    page_lens,
    th_in,
    m_buf,
    l_buf,
    o_buf,
    mask_buf,
    scale,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_p,
    stride_k_h,
    stride_k_t,
    stride_k_kp,
    stride_scale_b,
    stride_scale_h,
    stride_scale_k,
    stride_v_p,
    stride_v_h,
    stride_v_t,
    stride_v_d,
    stride_pc_b,
    stride_plen_p,
    stride_th_b,
    stride_th_h,
    stride_m_b,
    stride_m_h,
    stride_m_t,
    stride_l_b,
    stride_l_h,
    stride_l_t,
    stride_o_b,
    stride_o_h,
    stride_o_t,
    stride_o_d,
    stride_mask_b,
    stride_mask_h,
    stride_mask_t,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
):
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    page_count = tl.load(page_counts + pid_b * stride_pc_b)
    page_valid = pid_tb < page_count
    page_id = pid_tb.to(tl.int32)

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

    q_ptrs0 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k0[None, :] * stride_q_k
    q_ptrs1 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k1[None, :] * stride_q_k
    q_ptrs2 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k2[None, :] * stride_q_k
    q_ptrs3 = q + pid_b * stride_q_b + (base_hq + rows)[:, None] * stride_q_h + offs_k3[None, :] * stride_q_k
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k0 * stride_scale_k
    scale_ptrs1 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k1 * stride_scale_k
    scale_ptrs2 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k2 * stride_scale_k
    scale_ptrs3 = k_scale + pid_b * stride_scale_b + pid_hkv * stride_scale_h + offs_k3 * stride_scale_k
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

    th_rows = tl.load(th_in + pid_b * stride_th_b + (base_hq + rows) * stride_th_h, mask=row_mask, other=NEG_INF)

    offs_t = tl.arange(0, SBS)
    page_len = tl.load(page_lens + page_id * stride_plen_p, mask=page_valid, other=0)
    t_mask = page_valid & (offs_t < page_len)
    base_tok = page_id * stride_k_p + pid_hkv * stride_k_h + offs_t * stride_k_t
    tl.multiple_of(base_tok, K_PACKED)
    kq_ptrs = k_q + base_tok[None, :] + offs_kp[:, None] * stride_k_kp
    kq_packed = tl.load(kq_ptrs, mask=t_mask[None, :], other=0).to(tl.int32)
    kq0 = ((kq_packed >> 0) & QMAX).to(tl.float16)
    kq1 = ((kq_packed >> 2) & QMAX).to(tl.float16)
    kq2 = ((kq_packed >> 4) & QMAX).to(tl.float16)
    kq3 = ((kq_packed >> 6) & QMAX).to(tl.float16)
    b_s_q = tl.dot(q_scaled0, kq0, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled1, kq1, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled2, kq2, out_dtype=tl.float32)
    b_s_q += tl.dot(q_scaled3, kq3, out_dtype=tl.float32)
    b_s_q = (b_s_q + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s_act = tl.where(t_mask[None, :], b_s_q, NEG_INF)

    m_rows_blk = tl.max(b_s_act, axis=1)
    below = (m_rows_blk < th_rows) & row_mask
    n_below = tl.sum(below.to(tl.int32), axis=0)
    n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
    prune_blk = (n_below == n_valid) | (~page_valid)

    v_offs = tl.arange(0, V)
    if not prune_blk:
        b_p = tl.where(t_mask[None, :], tl.exp2(b_s_act - m_rows_blk[:, None]), 0.0)
        l_rows = tl.sum(b_p, axis=1)

        need_v = tl.sum(t_mask.to(tl.int32), axis=0) > 0
        o_tile = tl.zeros([BM_DOT, V], tl.float32)
        if need_v:
            v_ptrs = v + page_id * stride_v_p + pid_hkv * stride_v_h + offs_t[:, None] * stride_v_t + v_offs[None, :] * stride_v_d
            b_v = tl.load(v_ptrs, mask=t_mask[:, None] & (v_offs[None, :] < V), other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

        m_ptrs = m_buf + pid_b * stride_m_b + (base_hq + rows) * stride_m_h + pid_tb * stride_m_t
        l_ptrs = l_buf + pid_b * stride_l_b + (base_hq + rows) * stride_l_h + pid_tb * stride_l_t
        o_ptrs = (
            o_buf
            + pid_b * stride_o_b
            + (base_hq + rows)[:, None] * stride_o_h
            + pid_tb * stride_o_t
            + v_offs[None, :] * stride_o_d
        )
        tl.store(m_ptrs, m_rows_blk, mask=row_mask)
        tl.store(l_ptrs, l_rows, mask=row_mask)
        tl.store(o_ptrs, o_tile, mask=row_mask[:, None] & (v_offs[None, :] < V))

    mask_ptr = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h + pid_tb * stride_mask_t
    tl.store(mask_ptr, tl.where(prune_blk, tl.full((), 0, tl.int8), tl.full((), 1, tl.int8)))
@triton.jit
def paged_attn_stage2_masked(
    m_buf,
    l_buf,
    o_buf,
    mask_buf,
    o,
    stride_m_b,
    stride_m_h,
    stride_m_t,
    stride_l_b,
    stride_l_h,
    stride_l_t,
    stride_o_b,
    stride_o_h,
    stride_o_t,
    stride_o_d,
    stride_mask_b,
    stride_mask_h,
    stride_mask_t,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    NTB: tl.constexpr,
    B: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float("-inf"), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTB):
        keep = tl.load(mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h + tb * stride_mask_t).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_b * stride_m_b + pid_hq * stride_m_h + tb * stride_m_t)
            l_b = tl.load(l_buf + pid_b * stride_l_b + pid_hq * stride_l_h + tb * stride_l_t)
            o_b = tl.load(
                o_buf + pid_b * stride_o_b + pid_hq * stride_o_h + tb * stride_o_t + v_offs * stride_o_d
            )
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * stride_out_b + pid_hq * stride_out_h + v_offs * stride_out_d
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def _normalize_k_scale(k_scale: torch.Tensor, expect_shape):
    if k_scale.ndim == 2:
        k_scale = k_scale.unsqueeze(0)
    if k_scale.ndim != 3:
        raise ValueError(f"k_scale must be 2D or 3D, got {k_scale.ndim}D")
    if k_scale.shape != expect_shape:
        raise ValueError(f"Unsupported k_scale shape {k_scale.shape}, expected {expect_shape}")
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


class PagedAttnRunner:
    def __init__(
        self,
        q: torch.Tensor,  # [B, HQ, K]
        cache: PagedKVCache,
        *,
        delta: float = 5.0,
        precomputed_threshold: torch.Tensor | None = None,
        num_warps_th: int | None = None,
        num_stages_th: int | None = None,
        num_warps_s1: int | None = 4,
        num_stages_s1: int | None = 2,
        num_warps_s2: int | None = 2,
        num_stages_s2: int | None = 1,
    ) -> None:
        if q.dim() != 3:
            raise ValueError("q must be [B, HQ, K]")
        if not q.is_cuda:
            raise ValueError("q must be on CUDA")

        B, HQ, K = q.shape
        if B != 1:
            raise ValueError("PagedAttnRunner currently supports B=1")

        k_q = cache.k_q
        v = cache.v
        _, HKV, page_size, k_packed = k_q.shape
        _, _, _, V = v.shape
        if HQ % HKV != 0:
            raise ValueError(f"HQ must be divisible by HKV, got HQ={HQ}, HKV={HKV}")
        G = HQ // HKV

        vals_per_byte = 4
        expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
        if k_packed != expected_k_packed:
            raise ValueError("Packed K dimension does not match K for q2 packing")

        k_scale = _normalize_k_scale(cache.k_scale, (B, HKV, K))

        self.q = q.contiguous()
        self.cache = cache
        self.k_scale = k_scale
        self.B = B
        self.HQ = HQ
        self.HKV = HKV
        self.K = K
        self.V = V
        self.G = G
        self.SBS = page_size
        self.NTB = cache.block_table.shape[1]
        self.delta = float(delta)
        self.num_warps_th = num_warps_th
        self.num_stages_th = num_stages_th
        self.num_warps_s1 = num_warps_s1
        self.num_stages_s1 = num_stages_s1
        self.num_warps_s2 = num_warps_s2
        self.num_stages_s2 = num_stages_s2

        self.m_buf = torch.empty((B, HQ, self.NTB), device=q.device, dtype=torch.float32)
        self.l_buf = torch.empty((B, HQ, self.NTB), device=q.device, dtype=torch.float32)
        self.o_buf = torch.empty((B, HQ, self.NTB, V), device=q.device, dtype=torch.float32)
        self.mask_buf = torch.empty((B, HKV, self.NTB), device=q.device, dtype=torch.int8)
        self.o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)

        self.softmax_scale = 1.0 / math.sqrt(K)

        self.use_contig_pages = False
        if precomputed_threshold is not None:
            if precomputed_threshold.shape != (B, HQ):
                raise ValueError("precomputed_threshold must be [B, HQ]")
            self.threshold_buf = precomputed_threshold.contiguous()
        else:
            page_count = int(cache.page_counts[0].item())
            if page_count > 0:
                page_ids = cache.block_table[0, :page_count]
                contig_pages = torch.equal(
                    page_ids,
                    torch.arange(page_count, device=page_ids.device, dtype=page_ids.dtype),
                )
            else:
                contig_pages = False
            self.use_contig_pages = bool(contig_pages)

            self.threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
            th_kwargs = _kernel_kwargs(self.num_warps_th, self.num_stages_th)
            if self.use_contig_pages:
                paged_attn_compute_threshold_qbits_contig[(B, HKV)](
                    self.q,
                    k_q,
                    self.k_scale,
                    cache.page_counts,
                    cache.page_lens,
                    self.threshold_buf,
                    self.softmax_scale,
                    self.delta,
                    self.q.stride(0),
                    self.q.stride(1),
                    self.q.stride(2),
                    k_q.stride(0),
                    k_q.stride(1),
                    k_q.stride(2),
                    k_q.stride(3),
                    self.k_scale.stride(0),
                    self.k_scale.stride(1),
                    self.k_scale.stride(2),
                    cache.page_counts.stride(0),
                    cache.page_lens.stride(0),
                    self.threshold_buf.stride(0),
                    self.threshold_buf.stride(1),
                    B=self.B,
                    HKV=self.HKV,
                    HQ=self.HQ,
                    K=self.K,
                    K_PACKED=k_q.shape[-1],
                    G=self.G,
                    SBS=self.SBS,
                    **th_kwargs,
                )
            else:
                paged_attn_compute_threshold_qbits[(B, HKV)](
                    self.q,
                    k_q,
                    self.k_scale,
                    cache.block_table,
                    cache.page_counts,
                    cache.page_lens,
                    self.threshold_buf,
                    self.softmax_scale,
                    self.delta,
                    self.q.stride(0),
                    self.q.stride(1),
                    self.q.stride(2),
                    k_q.stride(0),
                    k_q.stride(1),
                    k_q.stride(2),
                    k_q.stride(3),
                    self.k_scale.stride(0),
                    self.k_scale.stride(1),
                    self.k_scale.stride(2),
                    cache.block_table.stride(0),
                    cache.block_table.stride(1),
                    cache.page_counts.stride(0),
                    cache.page_lens.stride(0),
                    self.threshold_buf.stride(0),
                    self.threshold_buf.stride(1),
                    B=self.B,
                    HKV=self.HKV,
                    HQ=self.HQ,
                    K=self.K,
                    K_PACKED=k_q.shape[-1],
                    G=self.G,
                    SBS=self.SBS,
                    **th_kwargs,
                )

    def run(self) -> torch.Tensor:
        q = self.q
        cache = self.cache
        k_q = cache.k_q
        v = cache.v

        s1_kwargs = _kernel_kwargs(self.num_warps_s1, self.num_stages_s1)
        grid = (self.NTB, self.B, self.HKV)
        if self.use_contig_pages:
            paged_attn_stage1_qbits_contig[grid](
                q,
                k_q,
                self.k_scale,
                v,
                cache.page_counts,
                cache.page_lens,
                self.threshold_buf,
                self.m_buf,
                self.l_buf,
                self.o_buf,
                self.mask_buf,
                self.softmax_scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k_q.stride(0),
                k_q.stride(1),
                k_q.stride(2),
                k_q.stride(3),
                self.k_scale.stride(0),
                self.k_scale.stride(1),
                self.k_scale.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                cache.page_counts.stride(0),
                cache.page_lens.stride(0),
                self.threshold_buf.stride(0),
                self.threshold_buf.stride(1),
                self.m_buf.stride(0),
                self.m_buf.stride(1),
                self.m_buf.stride(2),
                self.l_buf.stride(0),
                self.l_buf.stride(1),
                self.l_buf.stride(2),
                self.o_buf.stride(0),
                self.o_buf.stride(1),
                self.o_buf.stride(2),
                self.o_buf.stride(3),
                self.mask_buf.stride(0),
                self.mask_buf.stride(1),
                self.mask_buf.stride(2),
                B=self.B,
                HKV=self.HKV,
                HQ=self.HQ,
                K=self.K,
                K_PACKED=k_q.shape[-1],
                V=self.V,
                G=self.G,
                SBS=self.SBS,
                **s1_kwargs,
            )
        else:
            paged_attn_stage1_qbits[grid](
                q,
                k_q,
                self.k_scale,
                v,
                cache.block_table,
                cache.page_counts,
                cache.page_lens,
                self.threshold_buf,
                self.m_buf,
                self.l_buf,
                self.o_buf,
                self.mask_buf,
                self.softmax_scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k_q.stride(0),
                k_q.stride(1),
                k_q.stride(2),
                k_q.stride(3),
                self.k_scale.stride(0),
                self.k_scale.stride(1),
                self.k_scale.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                cache.block_table.stride(0),
                cache.block_table.stride(1),
                cache.page_counts.stride(0),
                cache.page_lens.stride(0),
                self.threshold_buf.stride(0),
                self.threshold_buf.stride(1),
                self.m_buf.stride(0),
                self.m_buf.stride(1),
                self.m_buf.stride(2),
                self.l_buf.stride(0),
                self.l_buf.stride(1),
                self.l_buf.stride(2),
                self.o_buf.stride(0),
                self.o_buf.stride(1),
                self.o_buf.stride(2),
                self.o_buf.stride(3),
                self.mask_buf.stride(0),
                self.mask_buf.stride(1),
                self.mask_buf.stride(2),
                B=self.B,
                HKV=self.HKV,
                HQ=self.HQ,
                K=self.K,
                K_PACKED=k_q.shape[-1],
                V=self.V,
                G=self.G,
                SBS=self.SBS,
                **s1_kwargs,
            )

        grid2 = (self.B, self.HKV, self.G)
        s2_kwargs = _kernel_kwargs(self.num_warps_s2, self.num_stages_s2)
        paged_attn_stage2_masked[grid2](
            self.m_buf,
            self.l_buf,
            self.o_buf,
            self.mask_buf,
            self.o,
            self.m_buf.stride(0),
            self.m_buf.stride(1),
            self.m_buf.stride(2),
            self.l_buf.stride(0),
            self.l_buf.stride(1),
            self.l_buf.stride(2),
            self.o_buf.stride(0),
            self.o_buf.stride(1),
            self.o_buf.stride(2),
            self.o_buf.stride(3),
            self.mask_buf.stride(0),
            self.mask_buf.stride(1),
            self.mask_buf.stride(2),
            self.o.stride(0),
            self.o.stride(1),
            self.o.stride(2),
            NTB=self.NTB,
            B=self.B,
            HKV=self.HKV,
            G=self.G,
            HQ=self.HQ,
            V=self.V,
            **s2_kwargs,
        )

        return self.o


def paged_attn_decode_q2(
    q: torch.Tensor,
    cache: PagedKVCache,
    *,
    delta: float = 5.0,
    num_warps_s1: int | None = 4,
    num_stages_s1: int | None = 2,
) -> torch.Tensor:
    runner = PagedAttnRunner(
        q,
        cache,
        delta=delta,
        num_warps_s1=num_warps_s1,
        num_stages_s1=num_stages_s1,
    )
    return runner.run()


__all__ = ["PagedAttnRunner", "paged_attn_decode_q2"]

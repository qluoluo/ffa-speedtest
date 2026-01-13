# CUDAGraph wrapper for Q1FP8 decode kernel (symmetric quantization).
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
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 1,
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
    offs_k4 = offs_kp * VALS_PER_BYTE + 4
    offs_k5 = offs_kp * VALS_PER_BYTE + 5
    offs_k6 = offs_kp * VALS_PER_BYTE + 6
    offs_k7 = offs_kp * VALS_PER_BYTE + 7
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K
    mask_k4 = offs_k4 < K
    mask_k5 = offs_k5 < K
    mask_k6 = offs_k6 < K
    mask_k7 = offs_k7 < K

    q_ptrs0 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k0[None, :]
    q_ptrs1 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k1[None, :]
    q_ptrs2 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k2[None, :]
    q_ptrs3 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k3[None, :]
    q_ptrs4 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k4[None, :]
    q_ptrs5 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k5[None, :]
    q_ptrs6 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k6[None, :]
    q_ptrs7 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k7[None, :]
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)
    q4 = tl.load(q_ptrs4, mask=row_mask[:, None] & mask_k4[None, :], other=0.0).to(tl.float16)
    q5 = tl.load(q_ptrs5, mask=row_mask[:, None] & mask_k5[None, :], other=0.0).to(tl.float16)
    q6 = tl.load(q_ptrs6, mask=row_mask[:, None] & mask_k6[None, :], other=0.0).to(tl.float16)
    q7 = tl.load(q_ptrs7, mask=row_mask[:, None] & mask_k7[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k0
    scale_ptrs1 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k1
    scale_ptrs2 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k2
    scale_ptrs3 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k3
    scale_ptrs4 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k4
    scale_ptrs5 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k5
    scale_ptrs6 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k6
    scale_ptrs7 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k7
    scale0 = tl.load(scale_ptrs0, mask=mask_k0, other=0.0).to(tl.float32)
    scale1 = tl.load(scale_ptrs1, mask=mask_k1, other=0.0).to(tl.float32)
    scale2 = tl.load(scale_ptrs2, mask=mask_k2, other=0.0).to(tl.float32)
    scale3 = tl.load(scale_ptrs3, mask=mask_k3, other=0.0).to(tl.float32)
    scale4 = tl.load(scale_ptrs4, mask=mask_k4, other=0.0).to(tl.float32)
    scale5 = tl.load(scale_ptrs5, mask=mask_k5, other=0.0).to(tl.float32)
    scale6 = tl.load(scale_ptrs6, mask=mask_k6, other=0.0).to(tl.float32)
    scale7 = tl.load(scale_ptrs7, mask=mask_k7, other=0.0).to(tl.float32)

    q_scaled0 = q0 * scale0[None, :].to(tl.float16)
    q_scaled1 = q1 * scale1[None, :].to(tl.float16)
    q_scaled2 = q2 * scale2[None, :].to(tl.float16)
    q_scaled3 = q3 * scale3[None, :].to(tl.float16)
    q_scaled4 = q4 * scale4[None, :].to(tl.float16)
    q_scaled5 = q5 * scale5[None, :].to(tl.float16)
    q_scaled6 = q6 * scale6[None, :].to(tl.float16)
    q_scaled7 = q7 * scale7[None, :].to(tl.float16)
    q_zero_sum = tl.sum(q_scaled0.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled1.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled2.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled3.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled4.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled5.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled6.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled7.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok0_q, K_PACKED)
    kq_ptrs0 = k_q + base_tok0_q[None, :] + offs_kp[:, None]
    kq_packed0 = tl.load(kq_ptrs0, mask=t_mask0[None, :], other=0).to(tl.int32)
    kq0_0 = ((kq_packed0 >> 0) & QMAX).to(tl.float16)
    kq0_1 = ((kq_packed0 >> 1) & QMAX).to(tl.float16)
    kq0_2 = ((kq_packed0 >> 2) & QMAX).to(tl.float16)
    kq0_3 = ((kq_packed0 >> 3) & QMAX).to(tl.float16)
    kq0_4 = ((kq_packed0 >> 4) & QMAX).to(tl.float16)
    kq0_5 = ((kq_packed0 >> 5) & QMAX).to(tl.float16)
    kq0_6 = ((kq_packed0 >> 6) & QMAX).to(tl.float16)
    kq0_7 = ((kq_packed0 >> 7) & QMAX).to(tl.float16)
    b_s0 = tl.dot(q_scaled0, kq0_0, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled1, kq0_1, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled2, kq0_2, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled3, kq0_3, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled4, kq0_4, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled5, kq0_5, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled6, kq0_6, out_dtype=tl.float32)
    b_s0 += tl.dot(q_scaled7, kq0_7, out_dtype=tl.float32)
    b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    tb1 = NTB - 1
    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    tl.multiple_of(base_tok1_q, K_PACKED)
    kq_ptrs1 = k_q + base_tok1_q[None, :] + offs_kp[:, None]
    kq_packed1 = tl.load(kq_ptrs1, mask=t_mask1[None, :], other=0).to(tl.int32)
    kq1_0 = ((kq_packed1 >> 0) & QMAX).to(tl.float16)
    kq1_1 = ((kq_packed1 >> 1) & QMAX).to(tl.float16)
    kq1_2 = ((kq_packed1 >> 2) & QMAX).to(tl.float16)
    kq1_3 = ((kq_packed1 >> 3) & QMAX).to(tl.float16)
    kq1_4 = ((kq_packed1 >> 4) & QMAX).to(tl.float16)
    kq1_5 = ((kq_packed1 >> 5) & QMAX).to(tl.float16)
    kq1_6 = ((kq_packed1 >> 6) & QMAX).to(tl.float16)
    kq1_7 = ((kq_packed1 >> 7) & QMAX).to(tl.float16)
    b_s1 = tl.dot(q_scaled0, kq1_0, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled1, kq1_1, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled2, kq1_2, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled3, kq1_3, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled4, kq1_4, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled5, kq1_5, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled6, kq1_6, out_dtype=tl.float32)
    b_s1 += tl.dot(q_scaled7, kq1_7, out_dtype=tl.float32)
    b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
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
    K_BITS: tl.constexpr = 1,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
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
    offs_k4 = offs_kp * VALS_PER_BYTE + 4
    offs_k5 = offs_kp * VALS_PER_BYTE + 5
    offs_k6 = offs_kp * VALS_PER_BYTE + 6
    offs_k7 = offs_kp * VALS_PER_BYTE + 7
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K
    mask_k4 = offs_k4 < K
    mask_k5 = offs_k5 < K
    mask_k6 = offs_k6 < K
    mask_k7 = offs_k7 < K

    q_ptrs0 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k0[None, :]
    q_ptrs1 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k1[None, :]
    q_ptrs2 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k2[None, :]
    q_ptrs3 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k3[None, :]
    q_ptrs4 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k4[None, :]
    q_ptrs5 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k5[None, :]
    q_ptrs6 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k6[None, :]
    q_ptrs7 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k7[None, :]
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)
    q4 = tl.load(q_ptrs4, mask=row_mask[:, None] & mask_k4[None, :], other=0.0).to(tl.float16)
    q5 = tl.load(q_ptrs5, mask=row_mask[:, None] & mask_k5[None, :], other=0.0).to(tl.float16)
    q6 = tl.load(q_ptrs6, mask=row_mask[:, None] & mask_k6[None, :], other=0.0).to(tl.float16)
    q7 = tl.load(q_ptrs7, mask=row_mask[:, None] & mask_k7[None, :], other=0.0).to(tl.float16)

    scale_ptrs0 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k0
    scale_ptrs1 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k1
    scale_ptrs2 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k2
    scale_ptrs3 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k3
    scale_ptrs4 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k4
    scale_ptrs5 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k5
    scale_ptrs6 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k6
    scale_ptrs7 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k7
    scale0 = tl.load(scale_ptrs0, mask=mask_k0, other=0.0).to(tl.float32)
    scale1 = tl.load(scale_ptrs1, mask=mask_k1, other=0.0).to(tl.float32)
    scale2 = tl.load(scale_ptrs2, mask=mask_k2, other=0.0).to(tl.float32)
    scale3 = tl.load(scale_ptrs3, mask=mask_k3, other=0.0).to(tl.float32)
    scale4 = tl.load(scale_ptrs4, mask=mask_k4, other=0.0).to(tl.float32)
    scale5 = tl.load(scale_ptrs5, mask=mask_k5, other=0.0).to(tl.float32)
    scale6 = tl.load(scale_ptrs6, mask=mask_k6, other=0.0).to(tl.float32)
    scale7 = tl.load(scale_ptrs7, mask=mask_k7, other=0.0).to(tl.float32)

    q_scaled0 = q0 * scale0[None, :].to(tl.float16)
    q_scaled1 = q1 * scale1[None, :].to(tl.float16)
    q_scaled2 = q2 * scale2[None, :].to(tl.float16)
    q_scaled3 = q3 * scale3[None, :].to(tl.float16)
    q_scaled4 = q4 * scale4[None, :].to(tl.float16)
    q_scaled5 = q5 * scale5[None, :].to(tl.float16)
    q_scaled6 = q6 * scale6[None, :].to(tl.float16)
    q_scaled7 = q7 * scale7[None, :].to(tl.float16)
    q_zero_sum = tl.sum(q_scaled0.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled1.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled2.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled3.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled4.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled5.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled6.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled7.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0_q = pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok0_q, K_PACKED)
        kq_ptrs0 = k_q + base_tok0_q[None, :] + offs_kp[:, None]
        kq_packed0 = tl.load(kq_ptrs0, mask=t_mask0[None, :], other=0).to(tl.int32)
        kq0_0 = ((kq_packed0 >> 0) & QMAX).to(tl.float16)
        kq0_1 = ((kq_packed0 >> 1) & QMAX).to(tl.float16)
        kq0_2 = ((kq_packed0 >> 2) & QMAX).to(tl.float16)
        kq0_3 = ((kq_packed0 >> 3) & QMAX).to(tl.float16)
        kq0_4 = ((kq_packed0 >> 4) & QMAX).to(tl.float16)
        kq0_5 = ((kq_packed0 >> 5) & QMAX).to(tl.float16)
        kq0_6 = ((kq_packed0 >> 6) & QMAX).to(tl.float16)
        kq0_7 = ((kq_packed0 >> 7) & QMAX).to(tl.float16)
        b_s0 = tl.dot(q_scaled0, kq0_0, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled1, kq0_1, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled2, kq0_2, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled3, kq0_3, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled4, kq0_4, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled5, kq0_5, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled6, kq0_6, out_dtype=tl.float32)
        b_s0 += tl.dot(q_scaled7, kq0_7, out_dtype=tl.float32)
        b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1_q = pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
        tl.multiple_of(base_tok1_q, K_PACKED)
        kq_ptrs1 = k_q + base_tok1_q[None, :] + offs_kp[:, None]
        kq_packed1 = tl.load(kq_ptrs1, mask=t_mask1[None, :], other=0).to(tl.int32)
        kq1_0 = ((kq_packed1 >> 0) & QMAX).to(tl.float16)
        kq1_1 = ((kq_packed1 >> 1) & QMAX).to(tl.float16)
        kq1_2 = ((kq_packed1 >> 2) & QMAX).to(tl.float16)
        kq1_3 = ((kq_packed1 >> 3) & QMAX).to(tl.float16)
        kq1_4 = ((kq_packed1 >> 4) & QMAX).to(tl.float16)
        kq1_5 = ((kq_packed1 >> 5) & QMAX).to(tl.float16)
        kq1_6 = ((kq_packed1 >> 6) & QMAX).to(tl.float16)
        kq1_7 = ((kq_packed1 >> 7) & QMAX).to(tl.float16)
        b_s1 = tl.dot(q_scaled0, kq1_0, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled1, kq1_1, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled2, kq1_2, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled3, kq1_3, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled4, kq1_4, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled5, kq1_5, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled6, kq1_6, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled7, kq1_7, out_dtype=tl.float32)
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
        kqsb1 = ((kq_packedsb >> 1) & QMAX).to(tl.float16)
        kqsb2 = ((kq_packedsb >> 2) & QMAX).to(tl.float16)
        kqsb3 = ((kq_packedsb >> 3) & QMAX).to(tl.float16)
        kqsb4 = ((kq_packedsb >> 4) & QMAX).to(tl.float16)
        kqsb5 = ((kq_packedsb >> 5) & QMAX).to(tl.float16)
        kqsb6 = ((kq_packedsb >> 6) & QMAX).to(tl.float16)
        kqsb7 = ((kq_packedsb >> 7) & QMAX).to(tl.float16)
        b_s_q = tl.dot(q_scaled0, kqsb0, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled1, kqsb1, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled2, kqsb2, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled3, kqsb3, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled4, kqsb4, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled5, kqsb5, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled6, kqsb6, out_dtype=tl.float32)
        b_s_q += tl.dot(q_scaled7, kqsb7, out_dtype=tl.float32)
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
                k_res_ptrs4 = k_res + base_toksb_k[None, :] + offs_k4[:, None]
                k_res_ptrs5 = k_res + base_toksb_k[None, :] + offs_k5[:, None]
                k_res_ptrs6 = k_res + base_toksb_k[None, :] + offs_k6[:, None]
                k_res_ptrs7 = k_res + base_toksb_k[None, :] + offs_k7[:, None]
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
                k_res4 = tl.load(
                    k_res_ptrs4,
                    mask=(mask_k4[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res5 = tl.load(
                    k_res_ptrs5,
                    mask=(mask_k5[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res6 = tl.load(
                    k_res_ptrs6,
                    mask=(mask_k6[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_res7 = tl.load(
                    k_res_ptrs7,
                    mask=(mask_k7[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                # Reuse selector b_s_q and add residual dot to avoid recomputing q·k_tile_q.
                b_s_res = tl.dot(q0, k_res0, out_dtype=tl.float32)
                b_s_res += tl.dot(q1, k_res1, out_dtype=tl.float32)
                b_s_res += tl.dot(q2, k_res2, out_dtype=tl.float32)
                b_s_res += tl.dot(q3, k_res3, out_dtype=tl.float32)
                b_s_res += tl.dot(q4, k_res4, out_dtype=tl.float32)
                b_s_res += tl.dot(q5, k_res5, out_dtype=tl.float32)
                b_s_res += tl.dot(q6, k_res6, out_dtype=tl.float32)
                b_s_res += tl.dot(q7, k_res7, out_dtype=tl.float32)
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


def _normalize_scale(k_scale: torch.Tensor, expect_shape):
    """
    Ensure scale tensors are contiguous and have shape [B, HKV, K].
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


def attn_forward_decode_quantized(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [B, T, HKV, ceil(K / (8 / k_bits))], packed quantized ints
    k_scale: torch.Tensor,     # [B, HKV, K] (token dimension removed)
    v: torch.Tensor,           # [B, T, HKV, V]
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K], fp8 residual
    k_bits: int = 1,
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
    return_kernel_timings: bool = False,
    **kwargs,
):
    # import os
    # print(f"ENTER {__file__} attn_forward_decode_quantized")
    
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 1:
        raise ValueError(f"attn_forward_decode_quantized currently supports 1-bit keys, got k_bits={k_bits}")
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
    k_scale = _normalize_scale(k_scale, expect_shape)

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
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
                K_BITS=k_bits,
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
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    s2_kwargs = _kernel_kwargs(num_warps_s2, num_stages_s2)
    def _launch_stage2():
        attn_forward_stage2_masked[(B, HKV, G)](
            m_buf, l_buf, o_buf,
            mask_buf,
            o, NTBS,
            B=B, HKV=HKV, G=G, HQ=HQ, V=V,
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

class CUDAGraphDecodeRunnerQ1FP8:
    """Capture and replay the Q1FP8 decode kernel with static buffers.

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
        k_bits: int = 1,
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
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
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

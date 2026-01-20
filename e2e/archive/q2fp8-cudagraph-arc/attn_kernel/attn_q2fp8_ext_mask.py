# Q2FP8 attention kernel with external mask (skip blocks based on pre-computed mask)
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def attn_forward_q2fp8_ext_mask_stage1(
    q, k_q, k_scale, k_res, v,
    m_buf, l_buf, o_buf,
    ext_mask,  # [B, HKV, NTBS] - external mask, 1=keep, 0=skip
    scale, T, NTB, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_FP8_RESIDUAL: tl.constexpr = False,
):
    """Stage1: Compute attention for kept blocks only (using external mask)."""
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

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
    offs_kp = tl.arange(0, K_PACKED)
    offs_k0 = offs_kp * VALS_PER_BYTE + 0
    offs_k1 = offs_kp * VALS_PER_BYTE + 1
    offs_k2 = offs_kp * VALS_PER_BYTE + 2
    offs_k3 = offs_kp * VALS_PER_BYTE + 3
    mask_k0 = offs_k0 < K
    mask_k1 = offs_k1 < K
    mask_k2 = offs_k2 < K
    mask_k3 = offs_k3 < K

    # Load q [BM_DOT, K] - split across packed dimensions
    q_ptrs0 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k0[None, :]
    q_ptrs1 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k1[None, :]
    q_ptrs2 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k2[None, :]
    q_ptrs3 = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k3[None, :]
    q0 = tl.load(q_ptrs0, mask=row_mask[:, None] & mask_k0[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(q_ptrs1, mask=row_mask[:, None] & mask_k1[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(q_ptrs2, mask=row_mask[:, None] & mask_k2[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(q_ptrs3, mask=row_mask[:, None] & mask_k3[None, :], other=0.0).to(tl.float16)

    # Load scale
    scale_ptrs0 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k0
    scale_ptrs1 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k1
    scale_ptrs2 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k2
    scale_ptrs3 = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k3
    scale0 = tl.load(scale_ptrs0, mask=mask_k0, other=0.0).to(tl.float32)
    scale1 = tl.load(scale_ptrs1, mask=mask_k1, other=0.0).to(tl.float32)
    scale2 = tl.load(scale_ptrs2, mask=mask_k2, other=0.0).to(tl.float32)
    scale3 = tl.load(scale_ptrs3, mask=mask_k3, other=0.0).to(tl.float32)

    # Pre-compute q_scaled and zero-point correction
    q_scaled0 = q0 * scale0[None, :].to(tl.float16)
    q_scaled1 = q1 * scale1[None, :].to(tl.float16)
    q_scaled2 = q2 * scale2[None, :].to(tl.float16)
    q_scaled3 = q3 * scale3[None, :].to(tl.float16)
    q_zero_sum = tl.sum(q_scaled0.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled1.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled2.to(tl.float32), axis=1)
    q_zero_sum += tl.sum(q_scaled3.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        tb_sb = pid_tb * NSB + sb

        # Check external mask - skip if mask says 0
        keep = tl.load(ext_mask + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb).to(tl.int1)
        if keep:
            base_toksb_q = pid_b * (T * HKV * K_PACKED) + offs_t_sb * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
            base_toksb_k = pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + (pid_hkv * K)
            tl.multiple_of(base_toksb_q, K_PACKED)
            tl.multiple_of(base_toksb_k, K)

            # Load and unpack quantized K
            kq_ptrssb = k_q + base_toksb_q[None, :] + offs_kp[:, None]
            kq_packedsb = tl.load(kq_ptrssb, mask=t_mask_sb[None, :], other=0).to(tl.int32)
            kqsb0 = ((kq_packedsb >> 0) & QMAX).to(tl.float16)
            kqsb1 = ((kq_packedsb >> 2) & QMAX).to(tl.float16)
            kqsb2 = ((kq_packedsb >> 4) & QMAX).to(tl.float16)
            kqsb3 = ((kq_packedsb >> 6) & QMAX).to(tl.float16)

            # Compute attention scores with quantized K
            b_s_q = tl.dot(q_scaled0, kqsb0, out_dtype=tl.float32)
            b_s_q += tl.dot(q_scaled1, kqsb1, out_dtype=tl.float32)
            b_s_q += tl.dot(q_scaled2, kqsb2, out_dtype=tl.float32)
            b_s_q += tl.dot(q_scaled3, kqsb3, out_dtype=tl.float32)
            b_s_q = (b_s_q + q_zero_sum[:, None]) * scale * RCP_LN2

            if USE_FP8_RESIDUAL:
                # Add FP8 residual contribution
                k_res_ptrs0 = k_res + base_toksb_k[None, :] + offs_k0[:, None]
                k_res_ptrs1 = k_res + base_toksb_k[None, :] + offs_k1[:, None]
                k_res_ptrs2 = k_res + base_toksb_k[None, :] + offs_k2[:, None]
                k_res_ptrs3 = k_res + base_toksb_k[None, :] + offs_k3[:, None]
                k_res0 = tl.load(k_res_ptrs0, mask=(mask_k0[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
                k_res1 = tl.load(k_res_ptrs1, mask=(mask_k1[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
                k_res2 = tl.load(k_res_ptrs2, mask=(mask_k2[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
                k_res3 = tl.load(k_res_ptrs3, mask=(mask_k3[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
                b_s_res = tl.dot(q0, k_res0, out_dtype=tl.float32)
                b_s_res += tl.dot(q1, k_res1, out_dtype=tl.float32)
                b_s_res += tl.dot(q2, k_res2, out_dtype=tl.float32)
                b_s_res += tl.dot(q3, k_res3, out_dtype=tl.float32)
                b_s_res = b_s_res * scale * RCP_LN2
                b_s = b_s_q + b_s_res
            else:
                b_s = b_s_q

            b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
            m_rows = tl.max(b_s, axis=1)

            b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            # Load V and compute output
            v_offs = tl.arange(0, V)
            v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            # Store intermediate results
            m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])


@triton.jit
def attn_forward_q2fp8_ext_mask_stage2(
    m_buf, l_buf, o_buf, ext_mask, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    """Stage2: Reduce across all kept blocks to get final output."""
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
        keep = tl.load(ext_mask + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
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


def attn_forward_q2fp8_with_ext_mask(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [B, T, HKV, K_packed]
    k_scale: torch.Tensor,     # [B, HKV, K]
    v: torch.Tensor,           # [B, T, HKV, V]
    ext_mask: torch.Tensor,    # [B, HKV, NTBS] - external mask, 1=keep, 0=skip
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K]
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    use_fp8_residual: bool = True,
):
    """Q2FP8 attention using external mask to skip blocks."""
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_bits != 2:
        raise ValueError(f"Only 2-bit keys supported, got k_bits={k_bits}")

    B, Tq, HQ, K = q.shape
    _, T, HKV, K_packed = k_q.shape
    _, _, _, V = v.shape
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # Normalize k_scale shape
    if k_scale.ndim == 4 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)
    k_scale = k_scale.contiguous()

    q = q.squeeze(1).contiguous()  # [B, HQ, K]
    k_q = k_q.contiguous()
    v = v.contiguous()

    use_fp8 = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8 else k_q

    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)

    attn_forward_q2fp8_ext_mask_stage1[(NTB, B, HKV)](
        q, k_q, k_scale, k_res, v,
        m_buf, l_buf, o_buf,
        ext_mask,
        scale, T, NTB, NTBS,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
        K_BITS=k_bits, USE_FP8_RESIDUAL=use_fp8,
    )

    attn_forward_q2fp8_ext_mask_stage2[(B, HKV, G)](
        m_buf, l_buf, o_buf,
        ext_mask,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
    )

    return o

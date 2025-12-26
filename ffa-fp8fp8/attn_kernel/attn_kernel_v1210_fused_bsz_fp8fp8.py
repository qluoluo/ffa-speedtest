# attn_kernel_v1210_fused_bsz_fp8fp8.py
# FP8 K with FP8 residual refinement.
import math

import torch
import triton
import triton.language as tl


@triton.jit
def attn_forward_stage1_fused_threshold_fp8(
    q,
    k_base,
    k_res,
    v,
    m_buf,
    l_buf,
    o_buf,
    mask_buf,
    scale,
    T,
    NTB,
    NTBS,
    delta,
    th_in,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K = tl.full([K], True, tl.int1)

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k = tl.arange(0, K)

    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0 = pid_b * (T * HKV * K) + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K)
        k_ptrs0 = k_base + base_tok0 + offs_k[:, None]
        k_tile0 = tl.load(k_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1 = pid_b * (T * HKV * K) + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K)
        k_ptrs1 = k_base + base_tok1 + offs_k[:, None]
        k_tile1 = tl.load(k_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb = pid_b * (T * HKV * K) + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K)
        k_ptrssb = k_base + base_toksb + offs_k[:, None]
        k_tile_base = tl.load(k_ptrssb, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
        b_s_base = tl.dot(q_tile, k_tile_base, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s_base, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            if USE_FP8_RESIDUAL:
                k_res_ptrssb = k_res + base_toksb + offs_k[:, None]
                k_res_tile = tl.load(
                    k_res_ptrssb,
                    mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_tile_refined = k_tile_base + k_res_tile
                b_s = tl.dot(q_tile, k_tile_refined, out_dtype=tl.float32) * scale * RCP_LN2
                b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
                m_rows = tl.max(b_s, axis=1)
            else:
                b_s = b_s_act
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
            tl.store(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf,
    l_buf,
    o_buf,
    mask_buf,
    o,
    NTBS,
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


def attn_forward_decode_fp8fp8(
    q: torch.Tensor,  # [B, 1, HQ, K]
    k_fp8: torch.Tensor,  # [B, T, HKV, K]
    v: torch.Tensor,  # [B, T, HKV, V]
    *,
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K]
    scale: float | None = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
    **kwargs,
):
    assert q.is_cuda and k_fp8.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if not k_fp8.is_floating_point():
        raise ValueError("k_fp8 must be a floating point tensor (fp8/fp16/bf16)")
    if k_residual is not None and not k_residual.is_floating_point():
        raise ValueError("k_residual must be a floating point tensor (fp8/fp16/bf16)")

    B, Tq, HQ, K = q.shape
    Bk, T, HKV, Kk = k_fp8.shape
    Bv, Tv, HKVv, V = v.shape
    if k_residual is not None:
        Bk_r, T_r, HKV_r, K_r = k_residual.shape
        assert (
            B == Bk == Bv == Bk_r
            and Tq == 1
            and Tv == T == T_r
            and HKVv == HKV == HKV_r
            and K == Kk == K_r
        ), "K/V layouts must be [B, T, HKV, D]"
    else:
        assert (
            B == Bk == Bv
            and Tq == 1
            and Tv == T
            and HKVv == HKV
            and K == Kk
        ), "K/V layouts must be [B, T, HKV, D]"

    G = HQ // HKV
    if HQ % HKV != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HKV ({HKV}) for GQA")

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    assert q.is_contiguous() and k_fp8.is_contiguous() and v.is_contiguous()
    if use_fp8_residual and k_residual is None:
        raise ValueError("use_fp8_residual=True requires k_residual")
    if k_residual is not None:
        assert k_residual.is_contiguous()

    q = q.contiguous()
    k_fp8 = k_fp8.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_fp8
    v = v.contiguous()

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
        use_ext_th = False

    attn_forward_stage1_fused_threshold_fp8[(NTB, B, HKV)](
        q,
        k_fp8,
        k_res,
        v,
        m_buf,
        l_buf,
        o_buf,
        mask_buf,
        scale,
        T,
        NTB,
        NTBS,
        delta,
        threshold_buf,
        B=B,
        HKV=HKV,
        HQ=HQ,
        K=K,
        V=V,
        G=G,
        BS=BS,
        SBS=SBS,
        USE_EXT_TH=use_ext_th,
        USE_FP8_RESIDUAL=use_fp8_residual,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    attn_forward_stage2_masked[(B, HKV, G)](
        m_buf,
        l_buf,
        o_buf,
        mask_buf,
        o,
        NTBS,
        B=B,
        HKV=HKV,
        G=G,
        HQ=HQ,
        V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    return o

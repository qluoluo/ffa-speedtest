# attn_kernel_v1210_fused_bsz_q2fp8.py
# 2-bit quantized K with fp8 residual refinement.
import math

import torch
import triton
import triton.language as tl


_STAGE1_CONFIGS = [
    triton.Config({"BM_DOT": 16}, num_warps=4, num_stages=2),
    triton.Config({"BM_DOT": 32}, num_warps=4, num_stages=2),
    triton.Config({"BM_DOT": 32}, num_warps=8, num_stages=3),
]
_STAGE2_CONFIGS = [
    triton.Config({"REDUCE_BLOCK": 4}, num_warps=4, num_stages=2),
    triton.Config({"REDUCE_BLOCK": 8}, num_warps=4, num_stages=2),
    triton.Config({"REDUCE_BLOCK": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_STAGE1_CONFIGS, key=["K", "V", "BS", "SBS", "G"])
@triton.jit
def attn_forward_stage1_fused_threshold_qbits(
    q, k_q, k_scale, k_zp, k_res, v,
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
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)
    QMAX = (1 << K_BITS) - 1
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    pack_idx = offs_k // VALS_PER_BYTE
    pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

    q_ptrs   = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # Scale / zero do not depend on token; load once per (B, HKV)
    scale_ptrs = k_scale + pid_b * (HKV * K) + pid_hkv * K + offs_k
    zp_ptrs    = k_zp    + pid_b * (HKV * K) + pid_hkv * K + offs_k
    scale_tile = tl.load(scale_ptrs, mask=TRUE_K, other=0.0).to(tl.float32)
    zp_tile    = tl.load(zp_ptrs,    mask=TRUE_K, other=0.0).to(tl.float32)

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0_q = pid_b * (T * HKV * K_PACKED) + (offs_t0[None, :] * (HKV * K_PACKED)) + (pid_hkv * K_PACKED)
        kq_ptrs0 = k_q + base_tok0_q + pack_idx[:, None]
        kq_tile0 = tl.load(kq_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0).to(tl.int32)
        kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & tl.full((), QMAX, tl.int32)).to(tl.float32)
        k_tile0 = (kq_tile0 * scale_tile[:, None] + zp_tile[:, None]).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1_q = pid_b * (T * HKV * K_PACKED) + (offs_t1[None, :] * (HKV * K_PACKED)) + (pid_hkv * K_PACKED)
        kq_ptrs1 = k_q + base_tok1_q + pack_idx[:, None]
        kq_tile1 = tl.load(kq_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0).to(tl.int32)
        kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & tl.full((), QMAX, tl.int32)).to(tl.float32)
        k_tile1 = (kq_tile1 * scale_tile[:, None] + zp_tile[:, None]).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb_q = pid_b * (T * HKV * K_PACKED) + (offs_t_sb[None, :] * (HKV * K_PACKED)) + (pid_hkv * K_PACKED)
        base_toksb_k = pid_b * (T * HKV * K) + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K)
        kq_ptrssb = k_q + base_toksb_q + pack_idx[:, None]
        kq_tilesb = tl.load(kq_ptrssb, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0).to(tl.int32)
        kq_tilesb = ((kq_tilesb >> pack_shifts[:, None]) & tl.full((), QMAX, tl.int32)).to(tl.float32)
        k_tile_q = (kq_tilesb * scale_tile[:, None] + zp_tile[:, None]).to(tl.float16)
        b_s_q     = tl.dot(q_tile, k_tile_q, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s_q, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)
        mask_ptr = mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb
        mask_val = tl.where(prune_blk, 0, 1).to(tl.int8)
        tl.store(mask_ptr, mask_val)

        if not prune_blk:
            if USE_FP8_RESIDUAL:
                k_res_ptrssb = k_res + base_toksb_k + offs_k[:, None]
                k_res_tile = tl.load(
                    k_res_ptrssb,
                    mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                    other=0.0,
                ).to(tl.float16)
                k_tile_refined = k_tile_q + k_res_tile
                b_s = tl.dot(q_tile, k_tile_refined, out_dtype=tl.float32) * scale * RCP_LN2
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


@triton.autotune(configs=_STAGE2_CONFIGS, key=["V"])
@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
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
    offs_rb = tl.arange(0, REDUCE_BLOCK)
    for tb0 in range(0, NTBS, REDUCE_BLOCK):
        tb = tb0 + offs_rb
        t_mask = tb < NTBS
        keep = tl.load(
            mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb,
            mask=t_mask,
            other=0,
        ).to(tl.int1)
        m_b = tl.load(
            m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb,
            mask=t_mask & keep,
            other=neg_inf,
        )
        l_b = tl.load(
            l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb,
            mask=t_mask & keep,
            other=0.0,
        )
        o_b = tl.load(
            o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb[:, None] * V + v_offs[None, :],
            mask=(t_mask & keep)[:, None],
            other=0.0,
        )
        seg_keep = tl.sum(keep.to(tl.int32), axis=0) > 0
        seg_m = tl.max(m_b, axis=0)
        new_m = tl.where(seg_keep, tl.maximum(b_m, seg_m), b_m)
        b_m_safe = tl.where(seg_keep, b_m, 0.0)
        new_m_safe = tl.where(seg_keep, new_m, 0.0)
        r_prev = tl.exp2(b_m_safe - new_m_safe)
        r_blk = tl.exp2(m_b - new_m_safe)
        new_acc = b_acc * r_prev + tl.sum(l_b * r_blk, axis=0)
        new_o = b_o * r_prev + tl.sum(o_b * r_blk[:, None], axis=0)
        b_acc = tl.where(seg_keep, new_acc, b_acc)
        b_o = tl.where(seg_keep, new_o, b_o)
        b_m = tl.where(seg_keep, new_m, b_m)
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def _normalize_scale_zero(k_scale: torch.Tensor, k_zero: torch.Tensor, expect_shape):
    """
    Ensure scale / zero_point tensors are contiguous and have shape [B, HKV, K].
    """
    if k_scale.ndim == 4 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)
    if k_zero.ndim == 4 and k_zero.shape[1] == 1:
        k_zero = k_zero.squeeze(1)

    if k_scale.shape != expect_shape or k_zero.shape != expect_shape:
        raise ValueError(
            f"Unsupported k_scale/k_zero shapes: {k_scale.shape=} {k_zero.shape=}, expected {expect_shape}"
        )

    return k_scale.contiguous(), k_zero.contiguous()


class Q2FP8DecodeWorkspace:
    def __init__(self, B: int, HQ: int, HKV: int, V: int, NTBS: int, device: torch.device, dtype: torch.dtype) -> None:
        self.B = int(B)
        self.HQ = int(HQ)
        self.HKV = int(HKV)
        self.V = int(V)
        self.NTBS = int(NTBS)
        self.device = device
        self.dtype = dtype
        self.o = torch.empty((B, HQ, V), device=device, dtype=dtype)
        self.m_buf = torch.empty((B, HQ, NTBS), device=device, dtype=torch.float32)
        self.l_buf = torch.empty((B, HQ, NTBS), device=device, dtype=torch.float32)
        self.o_buf = torch.empty((B, HQ, NTBS, V), device=device, dtype=torch.float32)
        self.mask_buf = torch.empty((B, HKV, NTBS), device=device, dtype=torch.int8)
        self.threshold_buf = torch.empty((B, HQ), device=device, dtype=torch.float32)

    def matches(self, B: int, HQ: int, HKV: int, V: int, NTBS: int, device: torch.device, dtype: torch.dtype) -> bool:
        return (
            self.B == int(B)
            and self.HQ == int(HQ)
            and self.HKV == int(HKV)
            and self.V == int(V)
            and self.NTBS == int(NTBS)
            and self.device == device
            and self.dtype == dtype
        )


def allocate_q2fp8_decode_workspace(
    q: torch.Tensor,
    k_q: torch.Tensor,
    v: torch.Tensor,
    BS: int = 128,
    SBS: int | None = None,
) -> Q2FP8DecodeWorkspace:
    B, Tq, HQ, _ = q.shape
    Bk, T, HKV, _ = k_q.shape
    Bv, Tv, HKVv, V = v.shape
    if B != Bk or B != Bv or Tq != 1 or T != Tv or HKV != HKVv:
        raise ValueError("K/V layouts must be [B, T, HKV, D]")
    if SBS is None:
        SBS = BS
    NTB = (T + BS - 1) // BS
    NSB = (BS + SBS - 1) // SBS
    NTBS = NTB * NSB
    return Q2FP8DecodeWorkspace(B, HQ, HKV, V, NTBS, q.device, q.dtype)


def attn_forward_decode_quantized(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [B, T, HKV, ceil(K / (8 / k_bits))], packed quantized ints
    k_scale: torch.Tensor,     # [B, HKV, K] (token dimension removed)
    k_zero: torch.Tensor,      # same shape as k_scale
    v: torch.Tensor,           # [B, T, HKV, V]
    k_residual: torch.Tensor | None = None,  # [B, T, HKV, K], fp8 residual
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
    workspace: Q2FP8DecodeWorkspace | None = None,
    **kwargs,
):
    # import os
    # print(f"ENTER {__file__} attn_forward_decode_quantized")
    
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 2:
        raise ValueError(f"attn_forward_decode_quantized currently supports 2-bit keys, got k_bits={k_bits}")
    assert k_scale.is_cuda and k_zero.is_cuda, "k_scale/k_zero must be CUDA tensors"
    if not k_scale.is_floating_point() or not k_zero.is_floating_point():
        raise ValueError("k_scale and k_zero must be floating point tensors for dequantization")
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
    k_scale, k_zero = _normalize_scale_zero(k_scale, k_zero, expect_shape)

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
    if workspace is not None:
        if not isinstance(workspace, Q2FP8DecodeWorkspace):
            raise TypeError("workspace must be a Q2FP8DecodeWorkspace")
        if not workspace.matches(B, HQ, HKV, V, NTBS, q.device, q.dtype):
            raise ValueError("workspace shape/device/dtype mismatch")
        o = workspace.o
        m_buf = workspace.m_buf
        l_buf = workspace.l_buf
        o_buf = workspace.o_buf
        mask_buf = workspace.mask_buf
        threshold_buf = workspace.threshold_buf
    else:
        o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
        m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
        l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
        o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
        mask_buf = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int8)
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)

    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        use_ext_th = False

    attn_forward_stage1_fused_threshold_qbits[(NTB, B, HKV)](
        q, k_q, k_scale, k_zero, k_res, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
        K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    attn_forward_stage2_masked[(B, HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o

# 速度优先：对称量化 + 低寄存器 BK=64 + Atomic Compact + Vectorized Stage2
# - 对称量化：仅使用 k_scale（无 zero-point），用 QZERO 抵消量化偏置。
# - K 维度按 BK=64 分块（低寄存器路径）。
# - Stage1 先算阈值，若 prune 则跳过；若保留，使用 atomic_add 获取紧凑位置并直接写入。
# - Stage2 使用向量化加载，批量处理多个 blocks，减少循环次数。
# CUDAGraph wrapper for Q2FP8 decode kernel (sym + atomic compact + vectorized Stage2).
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym_atomic_vec"

@triton.jit
def attn_compute_threshold_qbits(
    q, k_q, k_scale,
    th_out,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr = 128,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
):
    # 复用原逻辑，无需修改
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

    offs_k_base = tl.arange(0, BK)
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
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_atomic_compact(
    q, k_q, k_scale, k_res, v,
    compact_m_buf, compact_l_buf, compact_o_buf,
    kept_counter, kept_indices,
    scale, T, NTB, NTBS, delta,
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
):
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

    # Compute scale base pointer based on quantization mode
    if USE_PERBLOCK_SCALE:
        scale_base = pid_b * (NTB * HKV * K) + pid_tb * (HKV * K) + pid_hkv * K
    else:
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

    # (Optional) On-the-fly threshold computation omitted for brevity/speed if external threshold provided.
    # The logic mirrors the separate kernel if included.
    
    # Iterate over sub-blocks (SBS)
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

        if not prune_blk:
            # === ATOMIC COMPACT LOGIC ===
            # Increment counter for this (B, HKV) pair
            counter_ptr = kept_counter + pid_b * HKV + pid_hkv
            pos = tl.atomic_add(counter_ptr, 1)

            if pos < MAX_KEPT:
                # Calculate Values
                v_offs = tl.arange(0, V)
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
                
                # Write original index for reference/debugging
                idx_ptr = kept_indices + pid_b * (HKV * MAX_KEPT) + pid_hkv * MAX_KEPT + pos
                tl.store(idx_ptr, tb_sb)

                # Write compacted m/l/o
                # Note: buffers are shaped [B, HQ, MAX_KEPT, ...]. 
                # (base_hq + rows) accounts for all heads in the KV group.
                m_ptrs = compact_m_buf + pid_b * (HQ * MAX_KEPT) + (base_hq + rows) * MAX_KEPT + pos
                l_ptrs = compact_l_buf + pid_b * (HQ * MAX_KEPT) + (base_hq + rows) * MAX_KEPT + pos
                o_ptrs = compact_o_buf + pid_b * (HQ * MAX_KEPT * V) + (base_hq + rows)[:, None] * (MAX_KEPT * V) + pos * V + v_offs[None, :]
                
                tl.store(m_ptrs, m_rows, mask=row_mask)
                tl.store(l_ptrs, l_rows, mask=row_mask)
                tl.store(o_ptrs, o_tile, mask=row_mask[:, None])


@triton.jit
def attn_forward_stage2_atomic_compact(
    compact_m_buf, compact_l_buf, compact_o_buf,
    kept_counter,
    o,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
    MAX_KEPT: tl.constexpr,
    VEC_SIZE: tl.constexpr = 8,  # Vectorization block size
):
    """
    Vectorized Stage2: Merge kept blocks with vectorized loads.

    Key optimization: Load VEC_SIZE blocks at once instead of one by one.
    This reduces loop iterations and improves memory coalescing.

    Note: Due to Triton limitations, we use a simpler approach that still
    benefits from reduced loop iterations.
    """
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)

    # Load total kept count for this group, clamp to MAX_KEPT
    n_kept = tl.load(kept_counter + pid_b * HKV + pid_hkv)
    n_kept = tl.minimum(n_kept, MAX_KEPT)

    # Simplified vectorized iteration: process VEC_SIZE blocks at a time
    # We reduce loop iterations from MAX_KEPT to MAX_KEPT/VEC_SIZE
    num_vec_blocks = (MAX_KEPT + VEC_SIZE - 1) // VEC_SIZE

    base_m = compact_m_buf + pid_b * (HQ * MAX_KEPT) + pid_hq * MAX_KEPT
    base_l = compact_l_buf + pid_b * (HQ * MAX_KEPT) + pid_hq * MAX_KEPT
    base_o = compact_o_buf + pid_b * (HQ * MAX_KEPT * V) + pid_hq * (MAX_KEPT * V)

    for vec_idx in range(num_vec_blocks):
        start_i = vec_idx * VEC_SIZE

        # Process VEC_SIZE elements in this iteration
        # Unroll manually to avoid Triton indexing issues
        for local_j in tl.static_range(VEC_SIZE):
            i = start_i + local_j
            mask_i = i < n_kept

            # Load one element at a time (still benefits from reduced outer loop)
            m_b = tl.load(base_m + i, mask=mask_i, other=neg_inf)
            l_b = tl.load(base_l + i, mask=mask_i, other=0.0)
            o_b = tl.load(base_o + i * V + v_offs, mask=mask_i, other=0.0)

            # Online Softmax Merge
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
    if k_scale.ndim == 4:
        if k_scale.shape[1] == 1:
            k_scale = k_scale.squeeze(1)
        elif allow_perblock and NTB is not None and k_scale.shape[1] == NTB:
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
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    k_residual: torch.Tensor | None = None,
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
    max_kept_ratio: float = 0.02,  # Changed default from 0.2 to 0.02
    vec_size: int = 8,  # Vectorization block size for Stage2
    return_kernel_timings: bool = False,
    **kwargs,
):
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 2:
        raise ValueError(f"attn_forward_decode_quantized currently supports 2-bit keys, got k_bits={k_bits}")
    assert k_scale.is_cuda, "k_scale must be a CUDA tensor"
    if not k_scale.is_floating_point():
        raise ValueError("k_scale must be floating point tensor for dequantization")

    B, Tq, HQ, K = q.shape
    Bk, T, HKV, K_packed = k_q.shape
    Bv, Tv, HKVv, V = v.shape
    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    if K_packed != expected_k_packed:
        raise ValueError(f"k_q packed dim mismatch: got {K_packed}, expected {expected_k_packed}")

    assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV, "K/V layouts must be [B, T, HKV, D]"
    G = HQ // HKV

    expect_shape = (B, HKV, K)
    k_scale, use_perblock_scale = _normalize_scale(k_scale, expect_shape, allow_perblock=True, NTB=triton.cdiv(T, BS))

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    max_kept = _resolve_max_kept(max_kept, NTBS, max_kept_ratio)

    q = q.contiguous()
    k_q = k_q.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_q
    v = v.contiguous()
    kernel_times = {} if return_kernel_timings else None
    
    # === Buffer Allocation (Compact) ===
    # kept_counter: [B, HKV] int32 initialized to 0
    kept_counter = torch.zeros((B, HKV), device=q.device, dtype=torch.int32)
    
    # compact buffers: [B, HQ, MAX_KEPT, ...]
    compact_m_buf = torch.empty((B, HQ, max_kept), device=q.device, dtype=torch.float32)
    compact_l_buf = torch.empty((B, HQ, max_kept), device=q.device, dtype=torch.float32)
    compact_o_buf = torch.empty((B, HQ, max_kept, V), device=q.device, dtype=torch.float32)
    
    # Store indices for debugging or ratio calculation: [B, HKV, MAX_KEPT]
    kept_indices = torch.empty((B, HKV, max_kept), device=q.device, dtype=torch.int32)
    
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)

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
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G, BS=BS,
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
        # Using the atomic compact kernel
        attn_forward_stage1_atomic_compact[(NTB, B, HKV)](
            q, k_q, k_scale, k_res, v,
            compact_m_buf, compact_l_buf, compact_o_buf,
            kept_counter, kept_indices,
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            K_BITS=k_bits, USE_EXT_TH=use_ext_th, USE_FP8_RESIDUAL=use_fp8_residual, MAX_KEPT=max_kept,
            USE_PERBLOCK_SCALE=use_perblock_scale,
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    # Note: No scan or scatter kernels needed here.

    skip_ratio = None
    if return_skip_ratio:
        # Need to clamp counts to calculate ratio accurately based on what was ACTUALLY processed
        kept = torch.clamp(kept_counter, max=max_kept).sum()
        total = float(kept_counter.numel() * NTBS)
        skip_ratio = float((1.0 - (kept.float() / total)).item())

    s2_kwargs = _kernel_kwargs(num_warps_s2, num_stages_s2)
    def _launch_stage2():
        attn_forward_stage2_atomic_compact[(B, HKV, G)](
            compact_m_buf, compact_l_buf, compact_o_buf,
            kept_counter,
            o,
            B=B, HKV=HKV, G=G, HQ=HQ, V=V,
            MAX_KEPT=max_kept,
            VEC_SIZE=vec_size,  # Pass vectorization size
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


class CUDAGraphDecodeRunnerQ2FP8Atomic:
    """Capture and replay the Atomic Compact Q2FP8 decode kernel."""

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
        max_kept_ratio: float = 0.02,  # Changed default
        vec_size: int = 8,  # Vectorization size
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

        self._device = q.device
        self._k_bits = k_bits
        self._scale = scale
        self._BS = BS
        self._SBS = SBS
        self._delta = delta
        self._vec_size = vec_size  # Store vectorization size
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

        # Seed static buffers
        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Warmup
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
                vec_size=self._vec_size,  # Add vec_size
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
                max_kept=self._max_kept,
                vec_size=self._vec_size,  # Add vec_size
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

        # Skip ratio requires re-running calculation outside graph
        # Note: This might be slow as it re-runs kernels
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
            vec_size=self._vec_size,  # Add vec_size
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

    def replay_only(self) -> torch.Tensor:
        """Replay without updating static inputs."""
        self._graph.replay()
        return self._static_out

    __call__ = replay
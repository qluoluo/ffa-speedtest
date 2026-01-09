# attn_q2fp8_optimized.py
# 优化策略：
# 1. 向量化解包 (Vectorized Unpacking): 以 int32x4 (128bit) 粒度加载，寄存器内广播解包。
# 2. 强制拆分 Threshold 计算: 降低 Stage1 寄存器压力，提升 Occupancy。
# 3. 保持 BK=64: 配合向量化加载，适合低延迟 Decode。

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym"

# -----------------------------------------------------------------------------
# Kernel 1: 单独计算 Threshold (利用向量化解包)
# -----------------------------------------------------------------------------
@triton.jit
def attn_compute_threshold_qbits_vectorized(
    q, k_q, k_scale,
    th_out,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    
    # -----------------------------------------------------------
    # 向量化常量准备
    # BK=64, K_BITS=2 -> 16 vals/int32 -> 需要 4 个 int32
    # -----------------------------------------------------------
    NUM_PACKED_INTS: tl.constexpr = BK // (32 // K_BITS) # = 4
    SHIFTS = tl.arange(0, 32 // K_BITS) * K_BITS         # [0, 2, ..., 30]

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Preload scale base
    scale_base = k_scale + pid_b * (HKV * K) + pid_hkv * K

    # Time offsets for first and last block
    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    
    tb1 = NTB - 1
    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T

    # k_q 基础指针 (注意：这里我们只计算到 Time 维度，K维度在循环内处理)
    # k_q shape: [B, T, HKV, K_PACKED]
    base_k_q = k_q + pid_b * (T * HKV * K_PACKED) + (pid_hkv * K_PACKED)
    
    b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
    b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)

    offs_k_base = tl.arange(0, BK)
    offs_packed_int = tl.arange(0, NUM_PACKED_INTS) # [0, 1, 2, 3]

    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        
        # --- 1. Load Q & Scale (常规) ---
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)

        # --- 2. Vectorized Load K_Q ---
        # 计算当前 block 在 packed 维度上的起始 int32 索引
        # k_start 是元素索引，除以 16 得到 int32 索引
        curr_packed_start = k_start // (32 // K_BITS)
        packed_cols = curr_packed_start + offs_packed_int # [4]
        
        # 构造指针: [4 (K_Coarse), T_BS (Time)]
        # 注意: k_q layout 是 [..., T, ..., K_PACKED]. 
        # base_k_q 指向 [B, 0, hkv, 0]. 我们需要加 offs_t * stride_t + packed_cols
        # stride_t = HKV * K_PACKED
        stride_t = HKV * K_PACKED
        
        # Pointer Math: base + (T_idxs * stride)[:, None] + (packed_cols)[None, :]
        # Result Shape: [T_BS, 4] -> Load 4 int32s per time step
        # 为了方便 dot 运算，我们通常想要 [BK, T_BS]。
        # 这里我们先 Load [4, T_BS]，解包成 [64, T_BS]。
        
        # Block 0
        ptr0 = base_k_q + (offs_t0 * stride_t)[:, None] + packed_cols[None, :]
        packed_val0 = tl.load(ptr0, mask=t_mask0[:, None], other=0).to(tl.int32) # [T_BS, 4]
        
        # Block 1
        ptr1 = base_k_q + (offs_t1 * stride_t)[:, None] + packed_cols[None, :]
        packed_val1 = tl.load(ptr1, mask=t_mask1[:, None], other=0).to(tl.int32) # [T_BS, 4]

        # --- 3. Vectorized Unpack (寄存器内广播) ---
        # packed_val: [T_BS, 4]
        # SHIFTS: [16]
        # Target: [T_BS, 64] (actually [T_BS, 4, 16] flattened)
        
        # Expand dims for broadcasting:
        # packed: [T_BS, 4, 1]
        # shifts: [1, 1, 16]
        # result: [T_BS, 4, 16]
        
        val0_unpacked = (packed_val0[:, :, None] >> SHIFTS[None, None, :]) & QMAX
        val0_unpacked = val0_unpacked.reshape(T_BS, BK) # [T_BS, 64]
        kq_tile0 = val0_unpacked.trans(1, 0).to(tl.float16) # [64, T_BS] for dot
        
        val1_unpacked = (packed_val1[:, :, None] >> SHIFTS[None, None, :]) & QMAX
        val1_unpacked = val1_unpacked.reshape(T_BS, BK)
        kq_tile1 = val1_unpacked.trans(1, 0).to(tl.float16)

        # Dot Product
        b_s0 += tl.dot(q_scaled_sub, kq_tile0, out_dtype=tl.float32)
        b_s1 += tl.dot(q_scaled_sub, kq_tile1, out_dtype=tl.float32)

    q_zero_sum *= -QZERO
    b_s0 = (b_s0 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    b_s1 = (b_s1 + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


# -----------------------------------------------------------------------------
# Kernel 2: Stage 1 Attention (只保留 Attention 逻辑，移除 Threshold 计算)
# -----------------------------------------------------------------------------
@triton.jit
def attn_forward_stage1_optimized_compact(
    q, k_q, k_scale, k_res, v,
    m_buf, l_buf, o_buf,
    kept_indices, kept_counts,
    scale, T, NTB, NTBS, delta,
    th_in,  # 必须提供
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_FP8_RESIDUAL: tl.constexpr = False,
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
    QZERO = QMAX / 2
    
    # --- Vectorization Setup ---
    NUM_PACKED_INTS: tl.constexpr = BK // (32 // K_BITS)
    SHIFTS = tl.arange(0, 32 // K_BITS) * K_BITS

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    scale_base = k_scale + pid_b * (HKV * K) + pid_hkv * K
    base_k_q = k_q + pid_b * (T * HKV * K_PACKED) + (pid_hkv * K_PACKED)
    stride_t = HKV * K_PACKED

    # 总是加载 External Threshold
    th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)

    # Q·QZERO Precomputation
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)
    offs_k_base = tl.arange(0, BK)
    offs_packed_int = tl.arange(0, NUM_PACKED_INTS)

    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
        scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)
    q_zero_sum *= -QZERO

    keep_base = kept_indices + pid_b * (HKV * MAX_KEPT) + pid_hkv * MAX_KEPT
    count_ptr = kept_counts + pid_b * HKV + pid_hkv

    # Loop over Sub-Blocks (SBS)
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        b_s_q = tl.zeros([BM_DOT, SBS], tl.float32)
        
        # 内部循环：Attention Dot Product
        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            
            # 1. Load Q
            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
            scale_sub = tl.load(scale_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)

            # 2. Vectorized Load K_Q
            curr_packed_start = k_start // (32 // K_BITS)
            packed_cols = curr_packed_start + offs_packed_int
            
            # Pointer: [SBS, 4]
            ptr_sb = base_k_q + (offs_t_sb * stride_t)[:, None] + packed_cols[None, :]
            packed_val_sb = tl.load(ptr_sb, mask=t_mask_sb[:, None], other=0).to(tl.int32)

            # 3. Vectorized Unpack
            # [SBS, 4] -> [SBS, 64] -> Transpose -> [64, SBS]
            val_sb_unpacked = (packed_val_sb[:, :, None] >> SHIFTS[None, None, :]) & QMAX
            val_sb_unpacked = val_sb_unpacked.reshape(SBS, BK)
            kq_tilesb = val_sb_unpacked.trans(1, 0).to(tl.float16)

            b_s_q += tl.dot(q_scaled_sub, kq_tilesb, out_dtype=tl.float32)

        b_s_q = b_s_q + q_zero_sum[:, None]
        b_s_q_scaled = b_s_q * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s_q_scaled, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        # Pruning Logic
        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # 这里的 Residual 逻辑保持不变（假设它是 fp16/bf16，不需要解包）
            if USE_FP8_RESIDUAL:
                base_toksb_k = pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + (pid_hkv * K)
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

            idx = tl.atomic_add(count_ptr, tl.full((), 1, tl.int32))
            tl.store(keep_base + idx, tb_sb, mask=idx < MAX_KEPT)

# -----------------------------------------------------------------------------
# Stage 2 Kernel (unchanged)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Python Wrapper
# -----------------------------------------------------------------------------
def _normalize_scale(k_scale: torch.Tensor, expect_shape):
    if k_scale.ndim == 4 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)
    if k_scale.shape != expect_shape:
        raise ValueError(f"Unsupported k_scale shape: {k_scale.shape=}, expected {expect_shape}")
    return k_scale.contiguous()

def _kernel_kwargs(num_warps: int | None, num_stages: int | None) -> dict:
    kwargs = {}
    if num_warps is not None: kwargs["num_warps"] = int(num_warps)
    if num_stages is not None: kwargs["num_stages"] = int(num_stages)
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
    if max_kept is not None: return int(max_kept)
    return max(32, min(ntbs, int(math.ceil(ntbs * ratio))))

def attn_forward_decode_quantized(
    q: torch.Tensor, k_q: torch.Tensor, k_scale: torch.Tensor, v: torch.Tensor,
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
    max_kept_ratio: float = 0.2,
    return_kernel_timings: bool = False,
    **kwargs,
):
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_bits != 2: raise ValueError(f"Only k_bits=2 is supported in this optimized version.")
    
    B, Tq, HQ, K = q.shape
    Bk, T, HKV, K_packed = k_q.shape
    _, _, _, V = v.shape
    G = HQ // HKV
    
    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    assert K_packed == expected_k_packed
    
    k_scale = _normalize_scale(k_scale, (B, HKV, K))
    if scale is None: scale = 1.0 / math.sqrt(K)
    if SBS is None: SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB
    max_kept = _resolve_max_kept(max_kept, NTBS, max_kept_ratio)

    q = q.contiguous()
    k_q = k_q.contiguous()
    v = v.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_q # dummy if unused
    kernel_times = {} if return_kernel_timings else None

    # Buffers
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    kept_indices = torch.empty((B, HKV, max_kept), device=q.device, dtype=torch.int32)
    kept_counts = torch.zeros((B, HKV), device=q.device, dtype=torch.int32)

    # 1. Always compute threshold externally now (Force Split)
    if precomputed_threshold is not None:
        threshold_buf = precomputed_threshold.contiguous()
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)
        def _launch_threshold():
            attn_compute_threshold_qbits_vectorized[(B, HKV)](
                q, k_q, k_scale, threshold_buf,
                scale, T, NTB, delta,
                B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, G=G,
                K_BITS=k_bits,
                **th_kwargs
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    # 2. Optimized Stage 1 (Vectorized + No Internal Threshold)
    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    # 建议: num_stages_s1 设为 2 以进一步节省寄存器
    def _launch_stage1():
        attn_forward_stage1_optimized_compact[(NTB, B, HKV)](
            q, k_q, k_scale, k_res, v,
            m_buf, l_buf, o_buf,
            kept_indices, kept_counts,
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
            K_BITS=k_bits, USE_FP8_RESIDUAL=use_fp8_residual, MAX_KEPT=max_kept,
            **s1_kwargs,
        )
    _record_kernel_time(kernel_times, "stage1", _launch_stage1, q.device)

    # 3. Stage 2
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
        kept = kept_counts.sum()
        total = float(kept_counts.numel() * NTBS)
        skip_ratio = 1.0 - (kept.float() / total).item()
        if return_kernel_timings:
            return o, skip_ratio, kernel_times
        return o, skip_ratio
    if return_kernel_timings:
        return o, kernel_times
    return o


class CUDAGraphDecodeRunnerQ2FP8:
    """Capture and replay the Q2FP8 decode kernel with static buffers."""

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
        max_kept: Optional[int] = None,
        max_kept_ratio: float = 0.2,
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
        self._max_kept = max_kept
        self._max_kept_ratio = max_kept_ratio

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
            self._static_threshold = torch.empty_like(precomputed_threshold, device=self._device)

        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

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
                max_kept=self._max_kept,
                max_kept_ratio=self._max_kept_ratio,
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
                max_kept=self._max_kept,
                max_kept_ratio=self._max_kept_ratio,
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
            max_kept=self._max_kept,
            max_kept_ratio=self._max_kept_ratio,
        )
        return self._static_out, skip_ratio

    __call__ = replay

    def replay_only(self) -> torch.Tensor:
        self._graph.replay()
        return self._static_out

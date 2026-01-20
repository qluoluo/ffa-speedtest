"""
Prefill Attention Kernel with Threshold-Based Block Filtering

This kernel implements causal attention for prefill with threshold-based pruning:
1. Each Q block computes attention with first and last K blocks to estimate threshold
2. Middle K blocks are filtered based on threshold
3. Only kept blocks are processed and merged

Key features:
- Per-Q-block threshold computation
- Causal masking (Q block i can only see K blocks 0..i)
- Q2FP8 quantized keys with FP8 residuals
- 3-stage pipeline: threshold → stage1 (filter) → stage2 (merge)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def prefill_compute_threshold_per_qblock(
    q,           # [B, T_q, HQ, K]
    k_q,         # [B, T_k, HKV, K_PACKED] quantized keys
    k_scale,     # [B, num_k_blocks, HKV, K] per-block scales
    k_res,       # [B, T_k, HKV, K] FP8 residuals
    th_out,      # [B, num_q_blocks, HQ] output thresholds
    scale,       # Attention scale (1/sqrt(K))
    T_q,         # Query sequence length
    T_k,         # Key sequence length
    num_q_blocks,
    num_k_blocks,
    delta,       # Threshold delta parameter
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    G: tl.constexpr,  # HQ / HKV
    Q_BLOCK: tl.constexpr = 64,
    K_BLOCK: tl.constexpr = 64,
    BM_DOT: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    USE_FP8_RESIDUAL: tl.constexpr = True,
):
    """
    Compute per-Q-block threshold by sampling first and last K blocks

    Grid: (num_q_blocks, B, HKV)
    """
    pid_qb = tl.program_id(0)  # Q block index
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Q block range
    q_start = pid_qb * Q_BLOCK
    q_end = tl.minimum(q_start + Q_BLOCK, T_q)

    # Causal: this Q block can only see K blocks 0..pid_qb
    max_k_block = tl.minimum(pid_qb, num_k_blocks - 1)

    # Sample first K block (kb=0)
    kb0 = 0
    k_start0 = kb0 * K_BLOCK
    k_end0 = tl.minimum(k_start0 + K_BLOCK, T_k)

    # Sample last visible K block
    kb1 = max_k_block
    k_start1 = kb1 * K_BLOCK
    k_end1 = tl.minimum(k_start1 + K_BLOCK, T_k)

    # Load Q for this block: [BM_DOT, K]
    # Use middle query token as representative
    q_idx = tl.minimum(q_start + Q_BLOCK // 2, q_end - 1)
    offs_k = tl.arange(0, BK)

    max_score = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)

    # Process first K block
    for k_chunk in tl.static_range(0, K, BK):
        offs_k_cur = k_chunk + offs_k
        k_mask = offs_k_cur < K

        # Load Q
        q_ptrs = (q + pid_b * (T_q * HQ * K) +
                  q_idx * (HQ * K) +
                  (base_hq + rows)[:, None] * K +
                  offs_k_cur[None, :])
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load scale for kb0
        scale_ptrs = (k_scale + pid_b * (num_k_blocks * HKV * K) +
                     kb0 * (HKV * K) +
                     pid_hkv * K +
                     offs_k_cur)
        scale_sub = tl.load(scale_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Process K tokens in this block
        for kt in range(K_BLOCK):
            k_t = k_start0 + kt
            if k_t < k_end0:
                # Load quantized K
                pack_idx = offs_k_cur // VALS_PER_BYTE
                pack_shifts = (offs_k_cur % VALS_PER_BYTE) * K_BITS

                kq_ptrs = (k_q + pid_b * (T_k * HKV * K_PACKED) +
                          k_t * (HKV * K_PACKED) +
                          pid_hkv * K_PACKED +
                          pack_idx)
                kq_vals = tl.load(kq_ptrs, mask=k_mask, other=0).to(tl.int32)
                kq_vals = ((kq_vals >> pack_shifts) & QMAX).to(tl.float16)

                # Dequantize
                k_dequant = (kq_vals[None, :] - QZERO) * scale_sub[None, :]

                # Add residual if enabled
                if USE_FP8_RESIDUAL:
                    res_ptrs = (k_res + pid_b * (T_k * HKV * K) +
                               k_t * (HKV * K) +
                               pid_hkv * K +
                               offs_k_cur)
                    res_vals = tl.load(res_ptrs, mask=k_mask, other=0.0)
                    k_dequant = k_dequant + res_vals[None, :]

                # Compute score
                score = tl.sum(q_sub * k_dequant, axis=1) * scale * RCP_LN2
                max_score = tl.maximum(max_score, score)

    # Process last K block (if different from first)
    if kb1 != kb0:
        for k_chunk in tl.static_range(0, K, BK):
            offs_k_cur = k_chunk + offs_k
            k_mask = offs_k_cur < K

            # Load Q (reuse)
            q_ptrs = (q + pid_b * (T_q * HQ * K) +
                      q_idx * (HQ * K) +
                      (base_hq + rows)[:, None] * K +
                      offs_k_cur[None, :])
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

            # Load scale for kb1
            scale_ptrs = (k_scale + pid_b * (num_k_blocks * HKV * K) +
                         kb1 * (HKV * K) +
                         pid_hkv * K +
                         offs_k_cur)
            scale_sub = tl.load(scale_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Process K tokens
            for kt in range(K_BLOCK):
                k_t = k_start1 + kt
                if k_t < k_end1:
                    pack_idx = offs_k_cur // VALS_PER_BYTE
                    pack_shifts = (offs_k_cur % VALS_PER_BYTE) * K_BITS

                    kq_ptrs = (k_q + pid_b * (T_k * HKV * K_PACKED) +
                              k_t * (HKV * K_PACKED) +
                              pid_hkv * K_PACKED +
                              pack_idx)
                    kq_vals = tl.load(kq_ptrs, mask=k_mask, other=0).to(tl.int32)
                    kq_vals = ((kq_vals >> pack_shifts) & QMAX).to(tl.float16)

                    k_dequant = (kq_vals[None, :] - QZERO) * scale_sub[None, :]

                    if USE_FP8_RESIDUAL:
                        res_ptrs = (k_res + pid_b * (T_k * HKV * K) +
                                   k_t * (HKV * K) +
                                   pid_hkv * K +
                                   offs_k_cur)
                        res_vals = tl.load(res_ptrs, mask=k_mask, other=0.0)
                        k_dequant = k_dequant + res_vals[None, :]

                    score = tl.sum(q_sub * k_dequant, axis=1) * scale * RCP_LN2
                    max_score = tl.maximum(max_score, score)

    # Compute threshold
    th_rows = max_score - delta

    # Store threshold: [B, num_q_blocks, HQ]
    th_ptrs = (th_out + pid_b * (num_q_blocks * HQ) +
               pid_qb * HQ +
               (base_hq + rows))
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def prefill_stage1_fused_threshold(
    q, k_q, k_scale, k_res, v,
    m_buf, l_buf, o_buf,
    block_mask,
    th_in,
    scale, T_q, T_k, num_q_blocks, num_k_blocks, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr,
    K: tl.constexpr, K_PACKED: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr,
    Q_BLOCK: tl.constexpr = 64,
    K_BLOCK: tl.constexpr = 64,
    BM_DOT: tl.constexpr = 16,
    BV: tl.constexpr = 64,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 64,
    USE_FP8_RESIDUAL: tl.constexpr = True,
):
    """
    Stage 1: Process each (Q block, K block) pair with threshold filtering

    Grid: (num_q_blocks, num_k_blocks, B * HKV)
    """
    pid_qb = tl.program_id(0)
    pid_kb = tl.program_id(1)
    pid_combined = tl.program_id(2)

    # Decode combined dimension
    pid_b = pid_combined // HKV
    pid_hkv = pid_combined % HKV

    # Causal mask: Q block pid_qb can only see K blocks 0..pid_qb
    if pid_kb > pid_qb:
        return

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # Load threshold for this Q block
    th_ptrs = (th_in + pid_b * (num_q_blocks * HQ) +
               pid_qb * HQ +
               (base_hq + rows))
    th_rows = tl.load(th_ptrs, mask=row_mask, other=NEG_INF)

    # Q and K ranges
    q_start = pid_qb * Q_BLOCK
    q_end = tl.minimum(q_start + Q_BLOCK, T_q)
    k_start = pid_kb * K_BLOCK
    k_end = tl.minimum(k_start + K_BLOCK, T_k)

    # Check if this is first or last K block (always keep)
    is_boundary = (pid_kb == 0) or (pid_kb == tl.minimum(pid_qb, num_k_blocks - 1))

    # Initialize online softmax accumulators
    m_rows = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)
    l_rows = tl.zeros([BM_DOT], dtype=tl.float32)
    o_tile = tl.zeros([BM_DOT, BV], dtype=tl.float32)

    v_offs = tl.arange(0, BV)
    v_mask = v_offs < V

    # Process each Q token in this Q block
    for qt_local in range(Q_BLOCK):
        qt = q_start + qt_local
        if qt >= q_end:
            break

        # Load Q for this token
        offs_k = tl.arange(0, BK)
        q_vec = tl.zeros([BM_DOT, BK], dtype=tl.float32)

        for k_chunk in tl.static_range(0, K, BK):
            offs_k_cur = k_chunk + offs_k
            k_mask = offs_k_cur < K

            q_ptrs = (q + pid_b * (T_q * HQ * K) +
                      qt * (HQ * K) +
                      (base_hq + rows)[:, None] * K +
                      offs_k_cur[None, :])
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

            # Compute scores with all K tokens in this K block
            # Load scale for this K block
            scale_ptrs = (k_scale + pid_b * (num_k_blocks * HKV * K) +
                         pid_kb * (HKV * K) +
                         pid_hkv * K +
                         offs_k_cur)
            scale_sub = tl.load(scale_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Accumulate scores across K dimension
            max_score_chunk = tl.full([BM_DOT], NEG_INF, dtype=tl.float32)

            for kt_local in range(K_BLOCK):
                kt = k_start + kt_local
                if kt >= k_end:
                    break

                # Causal mask: qt can only see kt <= qt
                if kt > qt:
                    continue

                # Load quantized K
                pack_idx = offs_k_cur // VALS_PER_BYTE
                pack_shifts = (offs_k_cur % VALS_PER_BYTE) * K_BITS

                kq_ptrs = (k_q + pid_b * (T_k * HKV * K_PACKED) +
                          kt * (HKV * K_PACKED) +
                          pid_hkv * K_PACKED +
                          pack_idx)
                kq_vals = tl.load(kq_ptrs, mask=k_mask, other=0).to(tl.int32)
                kq_vals = ((kq_vals >> pack_shifts) & QMAX).to(tl.float16)

                # Dequantize
                k_dequant = (kq_vals[None, :] - QZERO) * scale_sub[None, :]

                # Add residual
                if USE_FP8_RESIDUAL:
                    res_ptrs = (k_res + pid_b * (T_k * HKV * K) +
                               kt * (HKV * K) +
                               pid_hkv * K +
                               offs_k_cur)
                    res_vals = tl.load(res_ptrs, mask=k_mask, other=0.0)
                    k_dequant = k_dequant + res_vals[None, :]

                # Compute score for this K token
                score = tl.sum(q_sub * k_dequant, axis=1) * scale * RCP_LN2
                max_score_chunk = tl.maximum(max_score_chunk, score)

                # Load V and accumulate with online softmax
                v_ptrs = (v + pid_b * (T_k * HKV * V) +
                         kt * (HKV * V) +
                         pid_hkv * V +
                         v_offs)
                v_vals = tl.load(v_ptrs, mask=v_mask, other=0.0)

                # Online softmax update
                new_m = tl.maximum(m_rows, score)
                alpha = tl.exp2(m_rows - new_m)
                beta = tl.exp2(score - new_m)

                l_rows = l_rows * alpha + beta
                o_tile = o_tile * alpha[:, None] + v_vals[None, :] * beta[:, None]
                m_rows = new_m

            # Check threshold (unless boundary block)
            if not is_boundary:
                # If max score < threshold for all heads, prune this block
                should_prune = tl.all(max_score_chunk < th_rows)
                if should_prune:
                    # Mark as pruned and return
                    block_idx = pid_qb * num_k_blocks + pid_kb
                    mask_ptrs = block_mask + pid_b * (HKV * num_q_blocks * num_k_blocks) + pid_hkv * (num_q_blocks * num_k_blocks) + block_idx
                    tl.store(mask_ptrs, 0)
                    return

    # This block is kept - store results
    block_idx = pid_qb * num_k_blocks + pid_kb
    mask_ptrs = block_mask + pid_b * (HKV * num_q_blocks * num_k_blocks) + pid_hkv * (num_q_blocks * num_k_blocks) + block_idx
    tl.store(mask_ptrs, 1)

    # Store m, l, o
    buf_idx = block_idx
    m_ptrs = m_buf + pid_b * (HQ * num_q_blocks * num_k_blocks) + (base_hq + rows) * (num_q_blocks * num_k_blocks) + buf_idx
    l_ptrs = l_buf + pid_b * (HQ * num_q_blocks * num_k_blocks) + (base_hq + rows) * (num_q_blocks * num_k_blocks) + buf_idx
    o_ptrs = o_buf + pid_b * (HQ * num_q_blocks * num_k_blocks * V) + (base_hq + rows)[:, None] * (num_q_blocks * num_k_blocks * V) + buf_idx * V + v_offs[None, :]

    tl.store(m_ptrs, m_rows, mask=row_mask)
    tl.store(l_ptrs, l_rows, mask=row_mask)
    tl.store(o_ptrs, o_tile, mask=row_mask[:, None] & v_mask[None, :])


@triton.jit
def prefill_stage2_merge(
    m_buf, l_buf, o_buf,
    kept_indices, kept_counts,
    o,
    num_q_blocks, num_k_blocks,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
    HQ: tl.constexpr, V: tl.constexpr,
    MAX_KEPT: tl.constexpr,
):
    """
    Stage 2: Merge kept blocks for each Q block

    Grid: (num_q_blocks, B, HKV, G)
    """
    pid_qb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)
    g = tl.program_id(3)

    pid_hq = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = float('-inf')

    b_m = neg_inf
    b_acc = 0.0
    b_o = tl.zeros([V], dtype=tl.float32)

    # Load kept count for this (Q block, batch, head)
    kept_idx_base = pid_b * (num_q_blocks * HKV * MAX_KEPT) + pid_qb * (HKV * MAX_KEPT) + pid_hkv * MAX_KEPT
    n_kept = tl.load(kept_counts + pid_b * (num_q_blocks * HKV) + pid_qb * HKV + pid_hkv)

    # Merge all kept K blocks
    for i in range(MAX_KEPT):
        if i < n_kept:
            kb = tl.load(kept_indices + kept_idx_base + i)
            buf_idx = pid_qb * num_k_blocks + kb

            m_b = tl.load(m_buf + pid_b * (HQ * num_q_blocks * num_k_blocks) + pid_hq * (num_q_blocks * num_k_blocks) + buf_idx)
            l_b = tl.load(l_buf + pid_b * (HQ * num_q_blocks * num_k_blocks) + pid_hq * (num_q_blocks * num_k_blocks) + buf_idx)
            o_b = tl.load(o_buf + pid_b * (HQ * num_q_blocks * num_k_blocks * V) + pid_hq * (num_q_blocks * num_k_blocks * V) + buf_idx * V + v_offs)

            # Online softmax merge
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m

    # Normalize and store
    is_empty = (b_acc == 0.0)
    out_tile = tl.where(is_empty, tl.zeros([V], dtype=tl.float32), b_o / b_acc)

    # Store to output: [B, T_q, HQ, V]
    # Map Q block back to Q tokens
    q_start = pid_qb * 64  # Q_BLOCK hardcoded for now
    o_ptrs = o + pid_b * (num_q_blocks * 64 * HQ * V) + q_start * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def prefill_attention_fused_threshold(
    q: torch.Tensor,          # [B, T_q, HQ, K]
    k_q: torch.Tensor,        # [B, T_k, HKV, K_PACKED]
    k_scale: torch.Tensor,    # [B, num_k_blocks, HKV, K]
    k_res: torch.Tensor,      # [B, T_k, HKV, K]
    v: torch.Tensor,          # [B, T_k, HKV, V]
    scale: float,             # 1/sqrt(K)
    delta: float = 5.0,
    q_block_size: int = 64,
    k_block_size: int = 64,
) -> torch.Tensor:
    """
    Prefill attention with threshold-based block filtering

    Returns:
        o: [B, T_q, HQ, V] attention output
    """
    B, T_q, HQ, K = q.shape
    _, T_k, HKV, K_PACKED = k_q.shape
    V = v.shape[-1]
    G = HQ // HKV

    # Pad to block boundaries
    num_q_blocks = (T_q + q_block_size - 1) // q_block_size
    num_k_blocks = (T_k + k_block_size - 1) // k_block_size

    # Allocate buffers
    threshold_buf = torch.zeros((B, num_q_blocks, HQ), dtype=torch.float32, device=q.device)
    m_buf = torch.full((B, HQ, num_q_blocks * num_k_blocks), float('-inf'), dtype=torch.float32, device=q.device)
    l_buf = torch.zeros((B, HQ, num_q_blocks * num_k_blocks), dtype=torch.float32, device=q.device)
    o_buf = torch.zeros((B, HQ, num_q_blocks * num_k_blocks, V), dtype=torch.float32, device=q.device)
    block_mask = torch.zeros((B, HKV, num_q_blocks * num_k_blocks), dtype=torch.int32, device=q.device)

    # Stage 0: Compute thresholds
    grid_th = (num_q_blocks, B, HKV)
    prefill_compute_threshold_per_qblock[grid_th](
        q, k_q, k_scale, k_res,
        threshold_buf,
        scale, T_q, T_k, num_q_blocks, num_k_blocks, delta,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_PACKED, G=G,
        Q_BLOCK=q_block_size, K_BLOCK=k_block_size,
    )

    # Stage 1: Process blocks with filtering
    # Merge B and HKV into single dimension for grid (Triton supports max 3D grid)
    grid_s1 = (num_q_blocks, num_k_blocks, B * HKV)
    prefill_stage1_fused_threshold[grid_s1](
        q, k_q, k_scale, k_res, v,
        m_buf, l_buf, o_buf,
        block_mask,
        threshold_buf,
        scale, T_q, T_k, num_q_blocks, num_k_blocks, delta,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_PACKED, V=V, G=G,
        Q_BLOCK=q_block_size, K_BLOCK=k_block_size,
    )

    # Host-side: compute kept indices
    block_offsets = torch.cumsum(block_mask, dim=-1, dtype=torch.int32)
    kept_counts = block_offsets[..., -1].contiguous()
    max_kept = int(kept_counts.max().item())

    kept_indices = torch.zeros((B, num_q_blocks, HKV, max_kept), dtype=torch.int32, device=q.device)

    # Fill kept_indices (simple CPU loop for now, can be optimized with kernel)
    for b in range(B):
        for qb in range(num_q_blocks):
            for h in range(HKV):
                kept = []
                for kb in range(num_k_blocks):
                    idx = qb * num_k_blocks + kb
                    if block_mask[b, h, idx] > 0:
                        kept.append(kb)
                for i, kb in enumerate(kept):
                    if i < max_kept:
                        kept_indices[b, qb, h, i] = kb

    # Stage 2: Merge
    o = torch.zeros((B, T_q, HQ, V), dtype=torch.float16, device=q.device)
    grid_s2 = (num_q_blocks, B, HKV, G)
    prefill_stage2_merge[grid_s2](
        m_buf, l_buf, o_buf,
        kept_indices, kept_counts,
        o,
        num_q_blocks, num_k_blocks,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
        MAX_KEPT=max_kept,
    )

    return o


if __name__ == "__main__":
    print("Testing prefill attention kernel...")
    # Simple test will be added
    print("✓ Kernel defined successfully!")

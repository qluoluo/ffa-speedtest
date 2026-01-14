# Sample4 + FP16 加速方案 (采样 K 不量化)
# - 每个 Block128 存储 4 个均匀采样点的 FP16 K 值用于快速筛选
# - 筛选阶段只用 4 个采样点计算近似分数（计算量 4/128 = 1/32）
# - 存储采样 K: FP16（无量化，存储量 4/128 = 1/32）
# - 保留完整 K/V cache 用于非剪枝 block 的精确计算
#
# 数据布局：
# - k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K] FP16 采样 K
# - k_full: [B, T, HKV, K] 完整 K cache
# - v: [B, T, HKV, V] 完整 V cache

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

# 采样点在 Block 内的偏移（均匀采样 4 个点）
# Block Size = 128, 采样间隔 = 32, 位置 = [0, 32, 64, 96]
SAMPLE_OFFSETS = [0, 32, 64, 96]
NUM_SAMPLES = 4


@triton.jit
def attn_compute_threshold_sample4_fp16(
    q, k_sample,
    th_out,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    BK: tl.constexpr = 64,
    NUM_SAMPLES: tl.constexpr = 4,
):
    """
    计算阈值：使用第一个和最后一个 block 的采样点。

    k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K] FP16
    """
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    # 第一个 block (tb0 = 0) 和最后一个 block (tb1 = NTB - 1)
    tb0 = 0
    tb1 = NTB - 1

    # 初始化累加器
    b_s0 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)
    b_s1 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

    offs_k_base = tl.arange(0, BK)
    sample_offs = tl.arange(0, NUM_SAMPLES)

    # 遍历 K 维度
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K

        # 加载 Q: [BM_DOT, BK]
        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        # 加载采样 K for block 0: [BK, NUM_SAMPLES]
        # k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K]
        k_base0 = k_sample + pid_b * (NTB * HKV * NUM_SAMPLES * K) + \
                  tb0 * (HKV * NUM_SAMPLES * K) + pid_hkv * (NUM_SAMPLES * K)
        k_ptrs0 = k_base0 + sample_offs[None, :] * K + offs_k[:, None]
        k_tile0 = tl.load(k_ptrs0, mask=k_mask[:, None], other=0.0).to(tl.float16)

        # 加载采样 K for block NTB-1: [BK, NUM_SAMPLES]
        k_base1 = k_sample + pid_b * (NTB * HKV * NUM_SAMPLES * K) + \
                  tb1 * (HKV * NUM_SAMPLES * K) + pid_hkv * (NUM_SAMPLES * K)
        k_ptrs1 = k_base1 + sample_offs[None, :] * K + offs_k[:, None]
        k_tile1 = tl.load(k_ptrs1, mask=k_mask[:, None], other=0.0).to(tl.float16)

        # 点积: [BM_DOT, BK] x [BK, NUM_SAMPLES] -> [BM_DOT, NUM_SAMPLES]
        b_s0 += tl.dot(q_sub, k_tile0, out_dtype=tl.float32)
        b_s1 += tl.dot(q_sub, k_tile1, out_dtype=tl.float32)

    # 应用 scale
    b_s0 = b_s0 * scale * RCP_LN2
    b_s1 = b_s1 * scale * RCP_LN2

    # 取每行的最大值
    m0 = tl.max(b_s0, axis=1)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_sample4_fp16(
    q, k_sample, k_full, v,
    m_buf, l_buf, o_buf,
    mask_buf, stride_mask_b, stride_mask_h, stride_mask_n,
    scale, T, NTB, NTBS, delta,
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,
    BK: tl.constexpr = 64,
    NUM_SAMPLES: tl.constexpr = 4,
):
    """
    Stage1: 使用 4 个 FP16 采样点快速筛选，非剪枝 block 使用完整 K 精确计算。

    k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K] - FP16 采样 K
    k_full: [B, T, HKV, K] - 完整 K cache
    """
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k_base = tl.arange(0, BK)
    sample_offs = tl.arange(0, NUM_SAMPLES)

    # 加载或计算阈值
    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        # 内联计算阈值
        tb0 = 0
        tb1 = NTB - 1
        b_s0 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)
        b_s1 = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            k_base0 = k_sample + pid_b * (NTB * HKV * NUM_SAMPLES * K) + \
                      tb0 * (HKV * NUM_SAMPLES * K) + pid_hkv * (NUM_SAMPLES * K)
            k_ptrs0 = k_base0 + sample_offs[None, :] * K + offs_k[:, None]
            k_tile0 = tl.load(k_ptrs0, mask=k_mask[:, None], other=0.0).to(tl.float16)

            k_base1 = k_sample + pid_b * (NTB * HKV * NUM_SAMPLES * K) + \
                      tb1 * (HKV * NUM_SAMPLES * K) + pid_hkv * (NUM_SAMPLES * K)
            k_ptrs1 = k_base1 + sample_offs[None, :] * K + offs_k[:, None]
            k_tile1 = tl.load(k_ptrs1, mask=k_mask[:, None], other=0.0).to(tl.float16)

            b_s0 += tl.dot(q_sub, k_tile0, out_dtype=tl.float32)
            b_s1 += tl.dot(q_sub, k_tile1, out_dtype=tl.float32)

        b_s0 = b_s0 * scale * RCP_LN2
        b_s1 = b_s1 * scale * RCP_LN2
        m0 = tl.max(b_s0, axis=1)
        m1 = tl.max(b_s1, axis=1)
        th_rows = tl.maximum(m0, m1) - delta

    mask_base = mask_buf + pid_b * stride_mask_b + pid_hkv * stride_mask_h

    # 遍历当前大 block 内的所有 sub-block
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        tb_sb = pid_tb * NSB + sb

        # ===== 使用 FP16 采样点快速筛选 =====
        b_s_sample = tl.zeros([BM_DOT, NUM_SAMPLES], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K

            q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            # 加载当前 block 的 FP16 采样 K
            k_base_sb = k_sample + pid_b * (NTB * HKV * NUM_SAMPLES * K) + \
                        pid_tb * (HKV * NUM_SAMPLES * K) + pid_hkv * (NUM_SAMPLES * K)
            k_ptrs_sb = k_base_sb + sample_offs[None, :] * K + offs_k[:, None]
            k_tile_sb = tl.load(k_ptrs_sb, mask=k_mask[:, None], other=0.0).to(tl.float16)

            b_s_sample += tl.dot(q_sub, k_tile_sb, out_dtype=tl.float32)

        b_s_sample = b_s_sample * scale * RCP_LN2

        # 取采样点的最大分数作为 block 近似分数
        m_rows_sample = tl.max(b_s_sample, axis=1)

        # 判断是否剪枝
        below = (m_rows_sample < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        keep_flag = tl.where(prune_blk, 0, 1).to(tl.int8)
        tl.store(mask_base + tb_sb * stride_mask_n, keep_flag)
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # ===== 非剪枝 block：使用完整 K 精确计算 =====
            b_s_full = tl.zeros([BM_DOT, SBS], tl.float32)

            for k_start in tl.static_range(0, K, BK):
                offs_k = k_start + offs_k_base
                k_mask = offs_k < K

                q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
                q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                # 加载完整 K: [K, SBS]
                k_base = k_full + pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + pid_hkv * K
                k_ptrs = k_base[None, :] + offs_k[:, None]
                k_tile = tl.load(k_ptrs, mask=k_mask[:, None] & t_mask_sb[None, :], other=0.0).to(tl.float16)

                b_s_full += tl.dot(q_sub, k_tile, out_dtype=tl.float32)

            b_s = b_s_full * scale * RCP_LN2
            b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
            m_rows = tl.max(b_s, axis=1)

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


def sample_k_fp16(
    k: torch.Tensor,
    BS: int = 128,
    sample_offsets: list = None,
) -> torch.Tensor:
    """
    提取 K 的采样点（不量化，保持 FP16）。

    Args:
        k: [B, T, HKV, K] 完整 K cache
        BS: Block Size，默认 128
        sample_offsets: 采样点在 block 内的偏移，默认 [0, 32, 64, 96]

    Returns:
        k_sample: [B, num_blocks, HKV, NUM_SAMPLES, K] FP16 采样 K
    """
    if sample_offsets is None:
        sample_offsets = SAMPLE_OFFSETS

    B, T, HKV, K = k.shape
    num_blocks = (T + BS - 1) // BS
    num_samples = len(sample_offsets)

    # Pad T to multiple of BS
    pad_T = num_blocks * BS - T
    if pad_T > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_T), value=0.0)

    # Reshape to [B, num_blocks, BS, HKV, K]
    k_blocks = k.view(B, num_blocks, BS, HKV, K)

    # 提取采样点: [B, num_blocks, num_samples, HKV, K]
    sample_indices = torch.tensor(sample_offsets, device=k.device, dtype=torch.long)
    k_samples = k_blocks[:, :, sample_indices, :, :]  # [B, num_blocks, num_samples, HKV, K]

    # 转置为 [B, num_blocks, HKV, num_samples, K]
    k_sample = k_samples.permute(0, 1, 3, 2, 4).contiguous()

    return k_sample


# 兼容性别名
def quantize_k_sample4_2bit_symmetric(
    k: torch.Tensor,
    BS: int = 128,
    sample_offsets: list = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    兼容性接口：返回 (k_sample, dummy_scale)。
    k_sample 是 FP16 采样 K，dummy_scale 是空的占位张量。
    """
    k_sample = sample_k_fp16(k, BS, sample_offsets)
    # 返回一个 dummy scale，形状与原版一致但不使用
    B, num_blocks, HKV, num_samples, K = k_sample.shape
    dummy_scale = torch.zeros((B, num_blocks, HKV, K), device=k.device, dtype=k.dtype)
    return k_sample, dummy_scale


def attn_forward_decode_sample4(
    q: torch.Tensor,              # [B, 1, HQ, K]
    k_sample_q: torch.Tensor,     # [B, num_blocks, HKV, NUM_SAMPLES, K] FP16 (兼容命名)
    k_sample_scale: torch.Tensor, # [B, num_blocks, HKV, K] (未使用，保留兼容性)
    k_full: torch.Tensor,         # [B, T, HKV, K]
    v: torch.Tensor,              # [B, T, HKV, V]
    k_bits: int = 2,              # 未使用，保留兼容性
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
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
    """
    Sample4 + FP16 (不量化) 加速的 decode attention。

    使用 4 个 FP16 采样点快速筛选 block，非剪枝 block 使用完整 K 精确计算。
    """
    # k_sample_q 实际上是 FP16 的采样 K（为兼容性保留原参数名）
    k_sample = k_sample_q

    assert q.is_cuda and k_sample.is_cuda and k_full.is_cuda and v.is_cuda

    B, Tq, HQ, K = q.shape
    Bk, num_blocks, HKV, num_samples, Ks = k_sample.shape
    _, T, _, _ = k_full.shape
    Bv, Tv, HKVv, V = v.shape

    assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV and K == Ks
    assert num_samples == NUM_SAMPLES, f"Expected {NUM_SAMPLES} samples, got {num_samples}"

    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    max_kept = _resolve_max_kept(max_kept, NTBS, max_kept_ratio)

    assert q.is_contiguous() and k_sample.is_contiguous() and k_full.is_contiguous() and v.is_contiguous()

    kernel_times = {} if return_kernel_timings else None
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    block_mask = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int8)
    block_offsets = torch.empty((B, HKV, NTBS), device=q.device, dtype=torch.int32)
    kept_indices = torch.empty((B, HKV, max_kept), device=q.device, dtype=torch.int32)
    kept_counts = torch.empty((B, HKV), device=q.device, dtype=torch.int32)

    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)
        def _launch_threshold():
            attn_compute_threshold_sample4_fp16[(B, HKV)](
                q, k_sample,
                threshold_buf,
                scale, T, NTB, delta,
                B=B, HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
                NUM_SAMPLES=num_samples,
                **th_kwargs,
            )
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True
    if kernel_times is not None and precomputed_threshold is not None:
        kernel_times["threshold"] = None

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)
    def _launch_stage1():
        attn_forward_stage1_sample4_fp16[(NTB, B, HKV)](
            q, k_sample, k_full, v,
            m_buf, l_buf, o_buf,
            block_mask, block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            scale, T, NTB, NTBS, delta,
            threshold_buf,
            B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
            USE_EXT_TH=use_ext_th, NUM_SAMPLES=num_samples,
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


# Alias for compatibility
attn_forward_decode_quantized = attn_forward_decode_sample4


class CUDAGraphDecodeRunnerSample4Q2:
    """Capture and replay the Sample4+FP16 decode kernel with static buffers."""

    def __init__(
        self,
        q: torch.Tensor,
        k_sample_q: torch.Tensor,
        k_sample_scale: torch.Tensor,
        k_full: torch.Tensor,
        v: torch.Tensor,
        *,
        precomputed_threshold: Optional[torch.Tensor] = None,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        max_kept: int | None = None,
        max_kept_ratio: float = 0.2,
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
        self._use_ext_th = precomputed_threshold is not None
        self._num_warps_th = num_warps_th
        self._num_stages_th = num_stages_th
        self._num_warps_s1 = num_warps_s1
        self._num_stages_s1 = num_stages_s1
        self._num_warps_s2 = num_warps_s2
        self._num_stages_s2 = num_stages_s2

        _, T, _, _ = k_full.shape
        sbs = BS if SBS is None else SBS
        ntb = triton.cdiv(T, BS)
        nsb = triton.cdiv(BS, sbs)
        ntbs = ntb * nsb
        self._max_kept = _resolve_max_kept(max_kept, ntbs, max_kept_ratio)

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_k_sample_q = torch.empty_like(k_sample_q, device=self._device)
        self._static_k_sample_scale = torch.empty_like(k_sample_scale, device=self._device)
        self._static_k_full = torch.empty_like(k_full, device=self._device)
        self._static_v = torch.empty_like(v, device=self._device)

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(
                precomputed_threshold, device=self._device
            )

        self._static_q.copy_(q)
        self._static_k_sample_q.copy_(k_sample_q)
        self._static_k_sample_scale.copy_(k_sample_scale)
        self._static_k_full.copy_(k_full)
        self._static_v.copy_(v)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        for _ in range(max(1, warmup)):
            attn_forward_decode_sample4(
                q=self._static_q,
                k_sample_q=self._static_k_sample_q,
                k_sample_scale=self._static_k_sample_scale,
                k_full=self._static_k_full,
                v=self._static_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
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
            self._static_out = attn_forward_decode_sample4(
                q=self._static_q,
                k_sample_q=self._static_k_sample_q,
                k_sample_scale=self._static_k_sample_scale,
                k_full=self._static_k_full,
                v=self._static_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
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
        k_sample_q: torch.Tensor,
        k_sample_scale: torch.Tensor,
        k_full: torch.Tensor,
        v: torch.Tensor,
        *,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")

        self._static_q.copy_(q)
        self._static_k_sample_q.copy_(k_sample_q)
        self._static_k_sample_scale.copy_(k_sample_scale)
        self._static_k_full.copy_(k_full)
        self._static_v.copy_(v)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        _, skip_ratio = attn_forward_decode_sample4(
            q=self._static_q,
            k_sample_q=self._static_k_sample_q,
            k_sample_scale=self._static_k_sample_scale,
            k_full=self._static_k_full,
            v=self._static_v,
            k_bits=self._k_bits,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            max_kept=self._max_kept,
            return_skip_ratio=True,
            precomputed_threshold=self._static_threshold,
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


# Alias for compatibility
CUDAGraphDecodeRunnerQ2FP8 = CUDAGraphDecodeRunnerSample4Q2

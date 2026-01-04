# Prototype: merge HQ heads into a larger GEMM for QK block maxima + threshold.
# This is intentionally separate from the main kernel implementation.
import math

import torch
import triton
import triton.language as tl


@triton.jit
def qk_block_max_hq_gemm(
    q, k_q, k_scale, k_zero,
    m_out,
    scale, T, NTB,
    B: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, K: tl.constexpr, K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    K_BITS: tl.constexpr = 2,
):
    # 2D grid = (B, NTB)
    pid_b = tl.program_id(0)
    pid_tb = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K = tl.full([K], True, tl.int1)
    QMAX = (1 << K_BITS) - 1
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < HQ
    offs_k = tl.arange(0, K)
    pack_idx = offs_k // VALS_PER_BYTE
    pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

    q_ptrs = q + pid_b * (HQ * K) + rows[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    t_start = pid_tb * BLOCK_N
    offs_t = t_start + tl.arange(0, BLOCK_N)
    t_mask = offs_t < T

    row_hkv = rows // G
    m_rows = tl.full([BLOCK_M], NEG_INF, tl.float32)

    for hkv in tl.static_range(0, HKV):
        scale_ptrs = k_scale + pid_b * (HKV * K) + hkv * K + offs_k
        zp_ptrs = k_zero + pid_b * (HKV * K) + hkv * K + offs_k
        scale_tile = tl.load(scale_ptrs, mask=TRUE_K, other=0.0).to(tl.float32)
        zp_tile = tl.load(zp_ptrs, mask=TRUE_K, other=0.0).to(tl.float32)

        base_tok_q = pid_b * (T * HKV * K_PACKED) + (offs_t[None, :] * (HKV * K_PACKED)) + (hkv * K_PACKED)
        kq_ptrs = k_q + base_tok_q + pack_idx[:, None]
        kq_tile = tl.load(kq_ptrs, mask=(TRUE_K[:, None] & t_mask[None, :]), other=0).to(tl.int32)
        kq_tile = ((kq_tile >> pack_shifts[:, None]) & tl.full((), QMAX, tl.int32)).to(tl.float32)
        k_tile = (kq_tile * scale_tile[:, None] + zp_tile[:, None]).to(tl.float16)

        scores = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        scores = tl.where(t_mask[None, :] & row_mask[:, None], scores, NEG_INF)
        max_rows = tl.max(scores, axis=1)

        m_rows = tl.where(row_hkv == hkv, max_rows, m_rows)

    m_ptrs = m_out + pid_b * (HQ * NTB) + rows * NTB + pid_tb
    tl.store(m_ptrs, m_rows, mask=row_mask)


def qk_block_max_hq_gemm_forward(
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    k_zero: torch.Tensor,
    *,
    scale: float | None = None,
    SBS: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """
    Compute per-block max logits for each HQ head using a merged-HQ GEMM path.

    q: [B, 1, HQ, K] or [B, HQ, K]
    k_q: [B, T, HKV, K_packed]
    k_scale/k_zero: [B, HKV, K]
    """
    if q.ndim == 4:
        q = q[:, 0]
    if q.ndim != 3:
        raise ValueError(f"q must be [B, HQ, K], got {q.shape}")
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if not k_q.is_cuda or not k_scale.is_cuda or not k_zero.is_cuda:
        raise ValueError("k_q/k_scale/k_zero must be CUDA tensors")

    B, HQ, K = q.shape
    Bk, T, HKV, K_packed = k_q.shape
    if Bk != B:
        raise ValueError(f"Batch mismatch: {B=} {Bk=}")
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    G = HQ // HKV
    block_m = 16 if HQ <= 16 else 32
    block_n = SBS
    NTB = triton.cdiv(T, block_n)

    q = q.contiguous()
    k_q = k_q.contiguous()
    k_scale = k_scale.contiguous()
    k_zero = k_zero.contiguous()

    m_out = torch.empty((B, HQ, NTB), device=q.device, dtype=torch.float32)

    grid = (B, NTB)
    qk_block_max_hq_gemm[grid](
        q, k_q, k_scale, k_zero,
        m_out,
        scale, T, NTB,
        B=B, HQ=HQ, HKV=HKV, K=K, K_PACKED=K_packed,
        G=G,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return m_out


def threshold_from_blocks(m_buf: torch.Tensor, delta: float):
    if m_buf.ndim != 3 or m_buf.shape[-1] < 1:
        raise ValueError(f"m_buf must be [B, HQ, NTB], got {m_buf.shape}")
    m0 = m_buf[:, :, 0]
    m1 = m_buf[:, :, -1]
    return torch.maximum(m0, m1) - float(delta)

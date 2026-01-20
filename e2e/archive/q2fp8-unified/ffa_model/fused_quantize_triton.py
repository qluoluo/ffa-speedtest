"""
Fused symmetric quantization kernel for Q2FP8 blocks.

This module is intentionally standalone and not wired into the cache yet.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _q2fp8_symmetric_quantize_kernel(
    k_ptr,
    k_q_ptr,
    k_scale_ptr,
    k_res_ptr,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kk,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_sb,
    stride_sblk,
    stride_sh,
    stride_sk,
    stride_rb,
    stride_rt,
    stride_rh,
    stride_rk,
    T,
    K,
    K_PACKED,
    N_K_BLOCKS,
    QZERO: tl.constexpr,
    QMAX: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    RES_DTYPE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    block_idx = pid0 // N_K_BLOCKS
    k_block_idx = pid0 - block_idx * N_K_BLOCKS

    offs_t = block_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < T

    k_start = k_block_idx * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K

    k_ptrs = (
        k_ptr
        + pid_b * stride_kb
        + offs_t[:, None] * stride_kt
        + pid_hkv * stride_kh
        + offs_k[None, :] * stride_kk
    )
    k = tl.load(k_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0)

    k_abs = tl.abs(k)
    k_abs = tl.where(t_mask[:, None] & k_mask[None, :], k_abs, 0.0)
    k_max = tl.max(k_abs, axis=0)
    scale = k_max / QZERO
    scale = tl.maximum(scale, EPS)

    scale_ptrs = (
        k_scale_ptr
        + pid_b * stride_sb
        + block_idx * stride_sblk
        + pid_hkv * stride_sh
        + offs_k[None, :] * stride_sk
    )
    tl.store(scale_ptrs, scale[None, :], mask=k_mask[None, :])

    scale_in = tl.cast(scale, IN_DTYPE)
    qzero = tl.full((1,), QZERO, IN_DTYPE)
    k_norm = tl.cast(k / scale_in[None, :], IN_DTYPE)
    q_in = k_norm + qzero
    q_f = tl.cast(q_in, tl.float32)
    q_floor = tl.floor(q_f)
    frac = q_f - q_floor
    is_half = frac == 0.5
    q_floor_i = tl.cast(q_floor, tl.int32)
    is_odd = (q_floor_i & 1) == 1
    round_up = frac > 0.5
    q_round = tl.where(round_up | (is_half & is_odd), q_floor + 1.0, q_floor)
    q_round = tl.minimum(tl.maximum(q_round, 0.0), QMAX)
    q_int = tl.cast(q_round, tl.int32)

    deq = (tl.cast(q_int, IN_DTYPE) - qzero) * scale_in[None, :]
    res = k - deq
    res = tl.cast(res, RES_DTYPE)

    res_ptrs = (
        k_res_ptr
        + pid_b * stride_rb
        + offs_t[:, None] * stride_rt
        + pid_hkv * stride_rh
        + offs_k[None, :] * stride_rk
    )
    tl.store(res_ptrs, res, mask=t_mask[:, None] & k_mask[None, :])

    q_int = tl.where(k_mask[None, :], q_int, 0)
    q_group = tl.reshape(q_int, (BLOCK_T, BLOCK_K // 4, 4))
    weights = tl.cast(1 << (2 * tl.arange(0, 4)), tl.int32)
    packed = tl.sum(q_group * weights, axis=2)
    packed = tl.cast(packed, tl.uint8)

    offs_kp = (k_start // 4) + tl.arange(0, BLOCK_K // 4)
    kp_mask = offs_kp < K_PACKED
    k_q_ptrs = (
        k_q_ptr
        + pid_b * stride_qb
        + offs_t[:, None] * stride_qt
        + pid_hkv * stride_qh
        + offs_kp[None, :] * stride_qk
    )
    tl.store(k_q_ptrs, packed, mask=t_mask[:, None] & kp_mask[None, :])


def quantize_symmetric_blocks_triton(
    k_blocks: torch.Tensor,
    block_size: int,
    k_bits: int = 2,
    eps: float = 1e-8,
    use_fp8_residual: bool = True,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not k_blocks.is_cuda:
        raise ValueError("Triton quantize kernel requires CUDA tensors.")
    if k_bits != 2:
        raise ValueError("Only 2-bit quantization is supported in this kernel.")
    if block_k % 4 != 0:
        raise ValueError("block_k must be a multiple of 4 for 2-bit packing.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not k_blocks.is_contiguous():
        k_blocks = k_blocks.contiguous()

    B, T, HKV, K = k_blocks.shape
    if T % block_size != 0:
        raise ValueError(f"T={T} is not divisible by block_size={block_size}")

    num_blocks = T // block_size
    k_packed = (K + 3) // 4

    k_q_packed = torch.empty((B, T, HKV, k_packed), device=k_blocks.device, dtype=torch.uint8)
    k_scale = torch.empty((B, num_blocks, HKV, K), device=k_blocks.device, dtype=k_blocks.dtype)

    residual_store_dtype = k_blocks.dtype
    if use_fp8_residual:
        residual_store_dtype = torch.float16
    k_residual = torch.empty((B, T, HKV, K), device=k_blocks.device, dtype=residual_store_dtype)

    grid = (num_blocks * triton.cdiv(K, block_k), B, HKV)

    _q2fp8_symmetric_quantize_kernel[grid](
        k_blocks,
        k_q_packed,
        k_scale,
        k_residual,
        k_blocks.stride(0),
        k_blocks.stride(1),
        k_blocks.stride(2),
        k_blocks.stride(3),
        k_q_packed.stride(0),
        k_q_packed.stride(1),
        k_q_packed.stride(2),
        k_q_packed.stride(3),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        k_scale.stride(3),
        k_residual.stride(0),
        k_residual.stride(1),
        k_residual.stride(2),
        k_residual.stride(3),
        T,
        K,
        k_packed,
        triton.cdiv(K, block_k),
        QZERO=1.5,
        QMAX=3.0,
        EPS=eps,
        BLOCK_T=block_size,
        BLOCK_K=block_k,
        IN_DTYPE=tl.float16 if k_blocks.dtype == torch.float16 else tl.bfloat16,
        RES_DTYPE=tl.float16 if residual_store_dtype == torch.float16 else tl.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if use_fp8_residual:
        if hasattr(torch, "float8_e4m3fn"):
            try:
                k_residual = k_residual.to(torch.float8_e4m3fn)
            except Exception:
                k_residual = k_residual.to(k_blocks.dtype)
        else:
            k_residual = k_residual.to(k_blocks.dtype)

    return k_q_packed, k_scale, k_residual

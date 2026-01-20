"""
Decode Attention Kernel Wrapper

This module provides a wrapper around the existing decode kernel from q2fp8-unified.
It reuses the proven decode implementation without modification.
"""

from __future__ import annotations
import sys
import os
from typing import Optional, Tuple

import torch

# Import from copied working kernels
try:
    from .attn_q2fp8_unified import (
        attn_compute_threshold_qbits,
        attn_forward_stage1_fused_threshold_qbits_compact,
        attn_forward_stage2_compact,
        attn_scatter_indices_kernel,
    )
    DECODE_KERNEL_AVAILABLE = True
except ImportError:
    try:
        from attn_q2fp8_unified import (
            attn_compute_threshold_qbits,
            attn_forward_stage1_fused_threshold_qbits_compact,
            attn_forward_stage2_compact,
            attn_scatter_indices_kernel,
        )
        DECODE_KERNEL_AVAILABLE = True
    except ImportError:
        DECODE_KERNEL_AVAILABLE = False
        print("Warning: Could not import decode kernel")


def decode_attention_fused_threshold(
    q: torch.Tensor,              # [B, 1, HQ, K] - single query token
    k_q: torch.Tensor,            # [B, T, HKV, K_PACKED] - quantized keys
    k_scale: torch.Tensor,        # [B, NTB, HKV, K] - per-block scales
    k_res: torch.Tensor,          # [B, T, HKV, K] - FP8 residuals
    v: torch.Tensor,              # [B, T, HKV, V] - values
    k_current: Optional[torch.Tensor] = None,  # [B, MAX_CURRENT, HKV, K] - FP16 current
    v_current: Optional[torch.Tensor] = None,  # [B, MAX_CURRENT, HKV, V]
    current_len: int = 0,
    scale: float = 1.0,
    delta: float = 5.0,
    block_size: int = 64,
    max_current: int = 128,
    use_perblock_scale: bool = True,
    use_fp8_residual: bool = True,
) -> torch.Tensor:
    """
    Decode attention using existing q2fp8-unified kernel

    Args:
        q: Query tensor [B, 1, HQ, K]
        k_q: Quantized keys [B, T, HKV, K_PACKED]
        k_scale: Per-block scales [B, NTB, HKV, K]
        k_res: FP8 residuals [B, T, HKV, K]
        v: Values [B, T, HKV, V]
        k_current: FP16 current keys (optional)
        v_current: FP16 current values (optional)
        current_len: Number of valid tokens in current buffer
        scale: Attention scale (1/sqrt(K))
        delta: Threshold delta parameter
        block_size: Block size for quantization
        max_current: Maximum current buffer size
        use_perblock_scale: Whether to use per-block scales
        use_fp8_residual: Whether to use FP8 residuals

    Returns:
        o: Attention output [B, HQ, V]
    """
    if not DECODE_KERNEL_AVAILABLE:
        raise RuntimeError("Decode kernel not available. Please check q2fp8-unified installation.")

    B, _, HQ, K = q.shape
    _, T, HKV, K_PACKED = k_q.shape
    V = v.shape[-1]
    G = HQ // HKV

    assert q.shape[1] == 1, "Decode mode requires single query token"

    # Reshape q to [B, HQ, K] for decode kernel
    q = q.squeeze(1).contiguous()

    # Compute number of token blocks
    NTB = (T + block_size - 1) // block_size
    NTBS = NTB + (1 if current_len > 0 else 0)  # +1 for current block if present

    # CRITICAL FIX: Ensure SBS is a power of 2 for Triton compatibility
    # tl.arange(0, SBS) requires SBS to be a power of 2
    SBS = block_size
    if SBS & (SBS - 1) != 0:  # Check if not power of 2
        # Round down to nearest power of 2
        SBS = 1 << (SBS.bit_length() - 1)
    # Ensure SBS is at least 1 and at most block_size
    SBS = max(1, min(SBS, block_size))

    # Allocate buffers
    threshold_buf = torch.zeros((B, HQ), dtype=torch.float32, device=q.device)
    m_buf = torch.full((B, HQ, NTBS), float('-inf'), dtype=torch.float32, device=q.device)
    l_buf = torch.zeros((B, HQ, NTBS), dtype=torch.float32, device=q.device)
    o_buf = torch.zeros((B, HQ, NTBS, V), dtype=torch.float32, device=q.device)
    block_mask = torch.zeros((B, HKV, NTBS), dtype=torch.int32, device=q.device)

    # Prepare current buffers
    if k_current is None:
        k_current = torch.zeros((B, max_current, HKV, K), dtype=torch.float16, device=q.device)
        v_current = torch.zeros((B, max_current, HKV, V), dtype=torch.float16, device=q.device)
        current_len = 0

    # Stage 0: Compute threshold
    grid_th = (B, HKV)
    attn_compute_threshold_qbits[grid_th](
        q, k_q, k_scale, k_current,
        threshold_buf,
        scale, T, NTB, delta, current_len,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_PACKED, G=G,
        BS=block_size,
        USE_PERBLOCK_SCALE=use_perblock_scale,
        MAX_CURRENT=max_current,
    )

    # Stage 1: Process blocks with threshold filtering
    grid_s1 = (NTB + 1, B, HKV)  # +1 for current block
    attn_forward_stage1_fused_threshold_qbits_compact[grid_s1](
        q, k_q, k_scale, k_res, v,
        k_current, v_current,
        m_buf, l_buf, o_buf,
        block_mask,
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
        scale, T, NTB, NTBS, delta, current_len,
        threshold_buf,
        B=B, HKV=HKV, HQ=HQ, K=K, K_PACKED=K_PACKED, V=V, G=G,
        BS=block_size, SBS=SBS,  # Use power-of-2 SBS
        USE_FP8_RESIDUAL=use_fp8_residual,
        USE_PERBLOCK_SCALE=use_perblock_scale,
        MAX_CURRENT=max_current,
    )

    # Host-side: compute kept indices
    block_offsets = torch.cumsum(block_mask, dim=-1, dtype=torch.int32)
    kept_counts = block_offsets[..., -1].contiguous()
    max_kept = int(kept_counts.max().item())
    max_kept = max(max_kept, 1)  # At least 1

    kept_indices = torch.zeros((B, HKV, max_kept), dtype=torch.int32, device=q.device)

    # Scatter kernel to fill kept_indices
    num_blocks_scatter = (NTBS + 255) // 256
    grid_scatter = (num_blocks_scatter, B, HKV)
    attn_scatter_indices_kernel[grid_scatter](
        block_mask, block_offsets, kept_indices,
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
        block_offsets.stride(0), block_offsets.stride(1), block_offsets.stride(2),
        kept_indices.stride(0), kept_indices.stride(1), kept_indices.stride(2),
        NTBS,
        MAX_KEPT=max_kept,
        BLOCK=256,
    )

    # Stage 2: Merge kept blocks
    o = torch.zeros((B, HQ, V), dtype=torch.float16, device=q.device)
    grid_s2 = (B, HKV, G)
    attn_forward_stage2_compact[grid_s2](
        m_buf, l_buf, o_buf,
        kept_indices, kept_counts,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
        MAX_KEPT=max_kept,
        HAS_CURRENT=(current_len > 0),
    )

    return o


if __name__ == "__main__":
    print("Testing decode attention wrapper...")
    if DECODE_KERNEL_AVAILABLE:
        print("✓ Decode kernel imported successfully!")
    else:
        print("✗ Decode kernel not available")

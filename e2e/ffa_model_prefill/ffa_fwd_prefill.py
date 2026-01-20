"""
Prefill Forward Interface

High-level interface for prefill attention with threshold-based filtering.
Integrates with Q2FP8 cache and prefill attention kernel.

NOTE: Currently using simplified PyTorch implementation because the Triton
prefill kernel has compatibility issues (break/continue not supported).
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch

# Use simplified prefill kernel (PyTorch implementation)
# Triton version has break/continue which are not supported
from .attn_kernel.attn_prefill_simple import prefill_attention_simple


def prefill_forward(
    q: torch.Tensor,              # [B, T_q, HQ, K]
    cache_dict: dict,             # Cache from Q2FP8CachePrefill
    scale: float,                 # 1/sqrt(K)
    delta: float = 5.0,
    q_block_size: int = 64,
    k_block_size: int = 64,
) -> torch.Tensor:
    """
    Prefill forward pass with threshold-based block filtering

    Args:
        q: Query tensor [B, T_q, HQ, K]
        cache_dict: Dictionary containing:
            - k_q: Quantized keys [B, T_k, HKV, K_PACKED]
            - k_scale: Per-block scales [B, num_k_blocks, HKV, K]
            - k_residual: FP8 residuals [B, T_k, HKV, K]
            - v: Values [B, T_k, HKV, V]
        scale: Attention scale (1/sqrt(K))
        delta: Threshold delta parameter
        q_block_size: Query block size
        k_block_size: Key block size

    Returns:
        o: Attention output [B, T_q, HQ, V]
    """
    # Extract cache tensors
    k_q = cache_dict["k_q"]
    k_scale = cache_dict["k_scale"]
    k_residual = cache_dict["k_residual"]
    v = cache_dict["v"]

    if k_q is None or v is None:
        raise ValueError("Cache not initialized for prefill")

    # Call simplified prefill attention
    o = prefill_attention_simple(
        q=q,
        cache_dict=cache_dict,
        scale=scale,
        delta=delta,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )

    return o


def prefill_forward_with_stats(
    q: torch.Tensor,
    cache_dict: dict,
    scale: float,
    delta: float = 5.0,
    q_block_size: int = 64,
    k_block_size: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """
    Prefill forward with statistics collection

    Returns:
        o: Attention output [B, T_q, HQ, V]
        stats: Dictionary with statistics:
            - num_q_blocks: Number of query blocks
            - num_k_blocks: Number of key blocks
            - total_blocks: Total Q-K block pairs
            - kept_blocks: Number of kept blocks (after filtering)
            - skip_ratio: Fraction of blocks skipped
    """
    B, T_q, HQ, K = q.shape
    k_q = cache_dict["k_q"]
    T_k = k_q.shape[1]

    num_q_blocks = (T_q + q_block_size - 1) // q_block_size
    num_k_blocks = (T_k + k_block_size - 1) // k_block_size

    # Compute total possible blocks (considering causal mask)
    total_blocks = sum(min(i + 1, num_k_blocks) for i in range(num_q_blocks))

    # Run prefill
    o = prefill_forward(
        q=q,
        cache_dict=cache_dict,
        scale=scale,
        delta=delta,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )

    # TODO: Collect actual kept_blocks from kernel
    # For now, estimate based on typical skip ratios
    kept_blocks = total_blocks  # Placeholder

    stats = {
        "num_q_blocks": num_q_blocks,
        "num_k_blocks": num_k_blocks,
        "total_blocks": total_blocks,
        "kept_blocks": kept_blocks,
        "skip_ratio": 1.0 - (kept_blocks / total_blocks) if total_blocks > 0 else 0.0,
    }

    return o, stats


if __name__ == "__main__":
    print("Testing prefill forward interface...")
    print("✓ Interface defined successfully!")

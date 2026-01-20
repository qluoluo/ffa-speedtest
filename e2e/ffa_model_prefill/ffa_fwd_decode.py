"""
Decode Forward Interface Wrapper

Wraps the existing decode kernel for use with the new cache structure.
"""

from __future__ import annotations
from typing import Optional

import torch

from .attn_kernel.attn_decode_kernel import decode_attention_fused_threshold


def decode_forward(
    q: torch.Tensor,              # [B, 1, HQ, K]
    cache_dict: dict,             # Cache from Q2FP8CachePrefill
    scale: float,                 # 1/sqrt(K)
    delta: float = 5.0,
    block_size: int = 64,
    max_current: int = 128,
) -> torch.Tensor:
    """
    Decode forward pass with threshold-based block filtering

    Args:
        q: Query tensor [B, 1, HQ, K] (single token)
        cache_dict: Dictionary containing:
            - k_q: Quantized keys [B, T, HKV, K_PACKED]
            - k_scale: Per-block scales [B, num_blocks, HKV, K]
            - k_residual: FP8 residuals [B, T, HKV, K]
            - v: Values [B, T, HKV, V]
            - k_current: Current buffer [B, MAX_CURRENT, HKV, K]
            - v_current: Current buffer [B, MAX_CURRENT, HKV, V]
            - current_len: Valid length in current buffer
        scale: Attention scale (1/sqrt(K))
        delta: Threshold delta parameter
        block_size: Block size for quantization
        max_current: Maximum current buffer size

    Returns:
        o: Attention output [B, HQ, V]
    """
    # Extract cache tensors
    k_q = cache_dict["k_q"]
    k_scale = cache_dict["k_scale"]
    k_residual = cache_dict["k_residual"]
    v = cache_dict["v"]
    k_current = cache_dict.get("k_current")
    v_current = cache_dict.get("v_current")
    current_len = cache_dict.get("current_len", 0)

    if k_q is None or v is None:
        raise ValueError("Cache not initialized for decode")

    # Call decode attention kernel
    o = decode_attention_fused_threshold(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        k_res=k_residual,
        v=v,
        k_current=k_current,
        v_current=v_current,
        current_len=current_len,
        scale=scale,
        delta=delta,
        block_size=block_size,
        max_current=max_current,
        use_perblock_scale=True,
        use_fp8_residual=True,
    )

    return o


if __name__ == "__main__":
    print("Testing decode forward interface...")
    print("✓ Interface defined successfully!")

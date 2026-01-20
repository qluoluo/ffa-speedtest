"""
Simplified Prefill Attention Kernel

This is a simplified version that uses PyTorch operations for prefill.
It still uses the quantized cache format but computes attention in PyTorch.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def prefill_attention_simple(
    q: torch.Tensor,              # [B, T_q, HQ, K]
    cache_dict: Dict[str, torch.Tensor],
    scale: float,
    delta: float = 5.0,
    q_block_size: int = 64,
    k_block_size: int = 64,
) -> torch.Tensor:
    """
    Simplified prefill attention using PyTorch operations

    Args:
        q: Query tensor [B, T_q, HQ, K]
        cache_dict: Dictionary containing:
            - k_q: Quantized keys [B, T_k, HKV, K_PACKED]
            - k_scale: Per-block scales [B, num_blocks, HKV, K]
            - k_residual: FP8 residuals [B, T_k, HKV, K]
            - v: Values [B, T_k, HKV, V]
        scale: Attention scale (1/sqrt(K))
        delta: Threshold delta (not used in simple version)
        q_block_size: Query block size (not used in simple version)
        k_block_size: Key block size for dequantization

    Returns:
        Output tensor [B, T_q, HQ, V]
    """
    B, T_q, HQ, K = q.shape

    # Get cache tensors
    k_q = cache_dict["k_q"]          # [B, T_k, HKV, K_PACKED]
    k_scale = cache_dict["k_scale"]  # [B, num_blocks, HKV, K]
    k_res = cache_dict["k_residual"] # [B, T_k, HKV, K]
    v = cache_dict["v"]              # [B, T_k, HKV, V]

    T_k = k_q.shape[1]
    HKV = k_q.shape[2]
    V = v.shape[-1]
    G = HQ // HKV  # Group size for GQA

    # Dequantize keys
    k_dequant = dequantize_keys(k_q, k_scale, k_res, k_block_size)  # [B, T_k, HKV, K]

    # Ensure everything is float16
    q = q.to(torch.float16)
    k_dequant = k_dequant.to(torch.float16)
    v = v.to(torch.float16)

    # Reshape for attention: [B, HQ, T_q, K] and [B, HKV, T_k, K]
    q = q.transpose(1, 2)  # [B, HQ, T_q, K]
    k_dequant = k_dequant.transpose(1, 2)  # [B, HKV, T_k, K]
    v = v.transpose(1, 2)  # [B, HKV, T_k, V]

    # Repeat KV for GQA
    if G > 1:
        k_dequant = k_dequant.repeat_interleave(G, dim=1)  # [B, HQ, T_k, K]
        v = v.repeat_interleave(G, dim=1)  # [B, HQ, T_k, V]

    # Compute attention scores
    attn_scores = torch.matmul(q, k_dequant.transpose(2, 3)) * scale  # [B, HQ, T_q, T_k]

    # Apply causal mask
    causal_mask = torch.triu(
        torch.ones(T_q, T_k, device=q.device, dtype=torch.bool),
        diagonal=T_k - T_q + 1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Softmax and attention
    attn_weights = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(attn_weights, v)  # [B, HQ, T_q, V]

    # Reshape back to [B, T_q, HQ, V]
    output = output.transpose(1, 2).contiguous()

    return output


def dequantize_keys(
    k_q: torch.Tensor,       # [B, T, HKV, K_PACKED]
    k_scale: torch.Tensor,   # [B, num_blocks, HKV, K]
    k_res: torch.Tensor,     # [B, T, HKV, K]
    block_size: int = 64,
    k_bits: int = 2,
) -> torch.Tensor:
    """
    Dequantize quantized keys

    Returns:
        k_dequant: [B, T, HKV, K]
    """
    B, T, HKV, K_PACKED = k_q.shape
    K = k_res.shape[-1]

    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2.0
    VALS_PER_BYTE = 8 // k_bits

    # Unpack quantized values
    k_q_int = k_q.to(torch.int32)  # [B, T, HKV, K_PACKED]

    # Expand to [B, T, HKV, K_PACKED, VALS_PER_BYTE]
    k_q_expanded = torch.zeros(B, T, HKV, K_PACKED, VALS_PER_BYTE, dtype=torch.int32, device=k_q.device)

    if k_bits == 2:
        for i in range(VALS_PER_BYTE):
            k_q_expanded[..., i] = (k_q_int >> (i * 2)) & 0x3
    else:  # 4-bit
        for i in range(VALS_PER_BYTE):
            k_q_expanded[..., i] = (k_q_int >> (i * 4)) & 0xF

    # Reshape to [B, T, HKV, K]
    k_q_unpacked = k_q_expanded.reshape(B, T, HKV, -1)[..., :K].to(torch.float16)

    # Get scales for each token
    num_blocks = T // block_size
    if T % block_size != 0:
        num_blocks += 1

    # Expand scales: [B, num_blocks, HKV, K] -> [B, T, HKV, K]
    k_scale_expanded = torch.zeros(B, T, HKV, K, dtype=torch.float16, device=k_scale.device)
    for b_idx in range(num_blocks):
        start_t = b_idx * block_size
        end_t = min((b_idx + 1) * block_size, T)
        k_scale_expanded[:, start_t:end_t] = k_scale[:, b_idx:b_idx+1].to(torch.float16)

    # Dequantize
    k_dequant = (k_q_unpacked - QZERO) * k_scale_expanded

    # Add residual
    k_dequant = k_dequant + k_res.to(torch.float16)

    return k_dequant


if __name__ == "__main__":
    print("Simplified prefill attention kernel loaded")

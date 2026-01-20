"""
FFA Model Prefill: Fast Forward Attention with Prefill Support

This package provides accelerated prefill and decode for LLMs using:
- Fused RoPE + quantization
- Threshold-based block filtering
- Q2FP8 quantized cache

Quick Start:
    from ffa_model_prefill import Q2FP8CachePrefill, LlamaAttentionPrefill
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig(use_ffa_prefill=True, use_ffa_decode=True)
    attn = LlamaAttentionPrefill(config, layer_idx=0).cuda()
    cache = Q2FP8CachePrefill(max_cache_len=8192, num_key_value_heads=8, head_dim=64)
"""

__version__ = "0.1.0"

from .q2fp8_cache_prefill import Q2FP8CachePrefill
from .modeling_llama_prefill import LlamaAttentionPrefill
from .ffa_fwd_prefill import prefill_forward, prefill_forward_with_stats
from .ffa_fwd_decode import decode_forward

__all__ = [
    "Q2FP8CachePrefill",
    "LlamaAttentionPrefill",
    "prefill_forward",
    "prefill_forward_with_stats",
    "decode_forward",
]

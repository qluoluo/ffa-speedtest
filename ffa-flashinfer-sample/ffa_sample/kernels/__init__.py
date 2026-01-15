"""
Triton kernels for sample-based sparse attention.
"""

from .triton_sample import (
    sample_k_fp16,
    attn_forward_decode_sample4,
    attn_compute_threshold_sample4_fp16,
    attn_forward_stage1_sample4_fp16,
    attn_forward_stage2_compact,
    SAMPLE_OFFSETS,
    NUM_SAMPLES,
)

__all__ = [
    "sample_k_fp16",
    "attn_forward_decode_sample4",
    "attn_compute_threshold_sample4_fp16",
    "attn_forward_stage1_sample4_fp16",
    "attn_forward_stage2_compact",
    "SAMPLE_OFFSETS",
    "NUM_SAMPLES",
]

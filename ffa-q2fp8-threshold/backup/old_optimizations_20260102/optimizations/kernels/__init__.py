"""
Optimized Q2FP8 attention kernels.

This package contains optimized versions of the Q2FP8 attention kernels with:
- LUT-based dequantization
- FP8 tensor core support (H100)
- Async memory copy (TMA on H100)
"""

from .attn_kernel_v1210_fused_bsz_q2fp8_optimized import (
    attn_forward_decode_quantized_optimized,
    create_dequant_lut,
)

from .attn_kernel_v1210_fused_bsz_q2fp8_optimized_cudagraph import (
    CUDAGraphDecodeRunnerQ2FP8Optimized,
)

__all__ = [
    'attn_forward_decode_quantized_optimized',
    'create_dequant_lut',
    'CUDAGraphDecodeRunnerQ2FP8Optimized',
]

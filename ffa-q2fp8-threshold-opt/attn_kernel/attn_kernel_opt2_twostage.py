"""
Optimization 2: Two-Stage Selector
- Coarse filter using sampled K dimension
- Fine selector only for survivors
- Expected speedup: 2-3x on selector computation
"""

# TODO: Full implementation
# For now, use Opt1 as placeholder
from .attn_kernel_opt1_compact import (
    CUDAGraphDecodeRunnerOpt1Compact as CUDAGraphDecodeRunnerOpt2TwoStage,
    attn_forward_decode_quantized,
)

__all__ = ["CUDAGraphDecodeRunnerOpt2TwoStage", "attn_forward_decode_quantized"]

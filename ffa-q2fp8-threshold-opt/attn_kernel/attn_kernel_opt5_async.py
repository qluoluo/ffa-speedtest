"""
Optimization 5: Async Pipeline
- Use H100 TMA for async memory copy
- Hide memory latency with computation
- Requires Triton async primitives
"""

# TODO: Full implementation (requires Triton async support)
# For now, use Opt1 as placeholder
from .attn_kernel_opt1_compact import (
    CUDAGraphDecodeRunnerOpt1Compact as CUDAGraphDecodeRunnerOpt5Async,
    attn_forward_decode_quantized,
)

__all__ = ["CUDAGraphDecodeRunnerOpt5Async", "attn_forward_decode_quantized"]

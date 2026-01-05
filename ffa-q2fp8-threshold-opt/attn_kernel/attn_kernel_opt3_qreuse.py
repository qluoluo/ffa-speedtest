"""
Optimization 3: Q Reuse
- Reorganize grid to (HQ, NTB)
- Each query head caches Q and iterates over T blocks
- Q load count: 16384 → 24
"""

# TODO: Full implementation
# For now, use Opt1 as placeholder
from .attn_kernel_opt1_compact import (
    CUDAGraphDecodeRunnerOpt1Compact as CUDAGraphDecodeRunnerOpt3QReuse,
    attn_forward_decode_quantized,
)

__all__ = ["CUDAGraphDecodeRunnerOpt3QReuse", "attn_forward_decode_quantized"]

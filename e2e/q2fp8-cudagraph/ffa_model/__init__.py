"""
FFA Model Package for Q2FP8 CUDA Graph

包含:
- Q2FP8CudaGraphCache: 预分配固定大小的 cache
- CudaGraphRunner: CUDA Graph 录制和重放
- attn_forward_decode: 统一的 decode attention 接口
"""

from .q2fp8_cudagraph_cache import Q2FP8CudaGraphCache, Q2FP8CudaGraphLayer
from .cudagraph_runner import CudaGraphRunner, MultiLengthCudaGraphRunner
from .ffa_fwd_decode import attn_forward_decode

__all__ = [
    "Q2FP8CudaGraphCache",
    "Q2FP8CudaGraphLayer",
    "CudaGraphRunner",
    "MultiLengthCudaGraphRunner",
    "attn_forward_decode",
]

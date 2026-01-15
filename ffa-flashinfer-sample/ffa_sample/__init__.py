"""
FFA-Sample: FlashInfer 集成的采样稀疏注意力

核心组件:
- SparseAttentionWithFlashInfer: 主要的注意力模块
- TritonSampleKernel: Triton 实现的采样筛选 kernel
- FlashInferWrapper: FlashInfer 的 Python 封装
"""

from .sparse_attention import SparseAttentionWithFlashInfer
from .kernels import (
    sample_k_fp16,
    attn_forward_decode_sample4,
    attn_compute_threshold_sample4_fp16,
)
from .utils import FlashInferDecodeWrapper, PagedKVCache

__all__ = [
    "SparseAttentionWithFlashInfer",
    "sample_k_fp16",
    "attn_forward_decode_sample4",
    "attn_compute_threshold_sample4_fp16",
    "FlashInferDecodeWrapper",
    "PagedKVCache",
]

__version__ = "0.1.0"

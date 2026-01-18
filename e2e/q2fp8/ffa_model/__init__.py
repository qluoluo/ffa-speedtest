"""
FFA-Q2FP8-Sym: 对称量化 K cache + FP8 残差

采用 Page-wise 量化策略：
- 按 block/page 独立量化，每个 block 有独立的 scale
- 新增 token 累积到当前 block，满了才量化
- 未满的 block 保持 FP16，不量化

支持 2-bit 和 4-bit 量化。
"""

from .oc_model import HF_ForCausalLM_FFA_Q2FP8_Sym_OC
from .modeling_llama import LlamaForCausalLM
from .q2fp8_cache import (
    Q2FP8SymCache,
    Q2FP8SymLayer,
    quantize_symmetric,
    quantize_block,
    SUPPORTED_K_BITS,
)
from .ffa_fwd_decode import attn_forward_decode

__all__ = [
    "HF_ForCausalLM_FFA_Q2FP8_Sym_OC",
    "LlamaForCausalLM",
    "Q2FP8SymCache",
    "Q2FP8SymLayer",
    "quantize_symmetric",
    "quantize_block",
    "SUPPORTED_K_BITS",
    "attn_forward_decode",
]

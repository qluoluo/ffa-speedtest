"""
FFA Q2FP8 Symmetric Decode Forward Interface

提供统一的 decode attention 接口，使用 Q2FP8 对称量化 + 阈值筛选方案。
支持 CUDA Graph 加速。
"""
import sys
import os

# 添加 attn_kernel 路径 (使用相对路径)
_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "..", "attn_kernel")
if _KERNEL_PATH not in sys.path:
    sys.path.insert(0, _KERNEL_PATH)

from attn_q2fp8_sym_mask import attn_forward_decode_quantized


def attn_forward_decode(
    *,
    q,
    k_q,
    k_scale,
    v,
    k_residual=None,
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    return_lse: bool = False,  # NEW: return (m, l) for accurate merging with k_current
    use_fp8_residual: bool = True,
    cudagraph_runner=None,  # NEW: CUDA Graph runner for acceleration
    **kwargs,
):
    """
    Q2FP8 Symmetric decode attention 接口。

    Args:
        q: [B, 1, HQ, K] Query tensor
        k_q: [B, T, HKV, K_packed] 2-bit 量化 K (packed uint8)
        k_scale: [B, HKV, K] (global) 或 [B, NTB, HKV, K] (per-block) 量化 scale
        v: [B, T, HKV, V] 完整 V cache
        k_residual: [B, T, HKV, K] FP8 残差 (可选)
        k_bits: 量化位数 (默认 2)
        scale: attention scale (默认 1/sqrt(K))
        BS: block size (默认 128)
        SBS: sub-block size (默认等于 BS)
        delta: 阈值偏移 (默认 5.0)
        return_skip_ratio: 是否返回跳过比例
        return_lse: 是否返回 (m, l) 用于与 k_current 准确合并
        use_fp8_residual: 是否使用 FP8 残差
        cudagraph_runner: CUDA Graph runner for kernel acceleration
        **kwargs: 其他参数

    Returns:
        if return_lse:
            attn_output: [B, HQ, V], final_m: [B, HQ], final_l: [B, HQ]
            (if return_skip_ratio: adds skip_ratio)
        else:
            attn_output: [B, HQ, V] Attention 输出
            skip_ratio (optional): 跳过的 block 比例
    """
    # 移除不需要的参数
    kwargs.pop("ffa_decode_kernel", None)
    kwargs.pop("k_sample", None)
    kwargs.pop("k_full", None)

    # Use CUDA Graph if available and conditions are met
    if cudagraph_runner is not None and not return_skip_ratio and not return_lse:
        # CUDA Graph path - fastest, but doesn't support skip_ratio or lse
        return cudagraph_runner.replay(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            k_zero=None,  # Not used in sym mode
            v=v,
            k_residual=k_residual,
        )
    else:
        # Standard path - supports all features
        return attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_residual=k_residual,
            k_bits=k_bits,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=return_skip_ratio,
            return_lse=return_lse,
            use_fp8_residual=use_fp8_residual,
            **kwargs,
        )

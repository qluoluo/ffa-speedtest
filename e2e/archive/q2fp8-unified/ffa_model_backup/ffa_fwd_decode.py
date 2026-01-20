"""
FFA Q2FP8 Unified Decode Forward Interface

提供统一的 decode attention 接口，使用 Q2FP8 对称量化 + unified kernel 处理 current tokens。
支持 CUDA Graph 加速。
"""
import sys
import os

# 添加 attn_kernel 路径 (使用相对路径)
_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "..", "attn_kernel")
if _KERNEL_PATH not in sys.path:
    sys.path.insert(0, _KERNEL_PATH)

from attn_q2fp8_unified import attn_forward_decode_quantized


def attn_forward_decode(
    *,
    q,
    k_q,
    k_scale,
    v,
    k_current=None,
    v_current=None,
    current_len: int = 0,
    k_residual=None,
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    use_fp8_residual: bool = True,
    cudagraph_runner=None,  # NEW: CUDA Graph runner for acceleration
    max_current: int = 128,
    **kwargs,
):
    """
    Q2FP8 Unified decode attention 接口。

    Args:
        q: [B, 1, HQ, K] Query tensor
        k_q: [B, T, HKV, K_packed] 2-bit 量化 K (packed uint8)
        k_scale: [B, HKV, K] (global) 或 [B, NTB, HKV, K] (per-block) 量化 scale
        v: [B, T, HKV, V] 完整 V cache
        k_current: [B, MAX_CURRENT, HKV, K] 当前未量化 tokens (固定大小 buffer)
        v_current: [B, MAX_CURRENT, HKV, V] 当前未量化 tokens 的 V
        current_len: 当前 buffer 有效长度
        k_residual: [B, T, HKV, K] FP8 残差 (可选)
        k_bits: 量化位数 (默认 2)
        scale: attention scale (默认 1/sqrt(K))
        BS: block size (默认 128)
        SBS: sub-block size (默认等于 BS)
        delta: 阈值偏移 (默认 5.0)
        return_skip_ratio: 是否返回跳过比例
        use_fp8_residual: 是否使用 FP8 残差
        cudagraph_runner: CUDA Graph runner for kernel acceleration
        max_current: current buffer 大小
        **kwargs: 其他参数

    Returns:
        attn_output: [B, HQ, V] Attention 输出
        skip_ratio (optional): 跳过的 block 比例
    """
    # 移除不需要的参数
    kwargs.pop("ffa_decode_kernel", None)
    kwargs.pop("k_sample", None)
    kwargs.pop("k_full", None)

    if kwargs.pop("return_lse", False):
        raise ValueError("Unified kernel does not support return_lse.")

    # Use CUDA Graph if available and conditions are met
    if cudagraph_runner is not None and not return_skip_ratio:
        # CUDA Graph path - fastest, skip_ratio requires extra kernel
        return cudagraph_runner.replay(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_current=k_current,
            v_current=v_current,
            current_len=current_len,
            k_residual=k_residual,
            return_skip_ratio=return_skip_ratio,
        )
    else:
        # Standard path - supports all features
        return attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_current=k_current,
            v_current=v_current,
            current_len=current_len,
            k_residual=k_residual,
            k_bits=k_bits,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=return_skip_ratio,
            use_fp8_residual=use_fp8_residual,
            max_current=max_current,
            **kwargs,
        )

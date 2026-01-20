"""
FFA Q2FP8 CUDA Graph Decode Forward Interface

提供统一的 decode attention 接口,使用 Q2FP8 对称量化 + CUDA Graph 加速。

核心特性:
1. Prefill 阶段使用 flash_attn
2. Decode 阶段使用 FFA + CUDA Graph
3. 预分配固定大小 buffer
4. 无动态内存分配
"""
import sys
import os

# 添加 attn_kernel 路径
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
    return_lse: bool = False,
    use_fp8_residual: bool = True,
    cudagraph_runner=None,
    **kwargs,
):
    """
    Q2FP8 CUDA Graph decode attention 接口。

    Args:
        q: [B, 1, HQ, K] Query tensor
        k_q: [B, T, HKV, K_packed] 2-bit 量化 K (packed uint8)
        k_scale: [B, NTB, HKV, K] per-block 量化 scale
        v: [B, T, HKV, V] 完整 V cache
        k_residual: [B, T, HKV, K] FP8 残差 (可选)
        k_bits: 量化位数 (默认 2)
        scale: attention scale (默认 1/sqrt(K))
        BS: block size (默认 128)
        SBS: sub-block size (默认等于 BS)
        delta: 阈值偏移 (默认 5.0)
        return_skip_ratio: 是否返回跳过比例
        return_lse: 是否返回 (m, l) 用于合并
        use_fp8_residual: 是否使用 FP8 残差
        cudagraph_runner: CUDA Graph runner for kernel acceleration
        **kwargs: 其他参数

    Returns:
        attn_output: [B, HQ, V] Attention 输出
        (可选) skip_ratio, m, l
    """
    # 移除不需要的参数
    kwargs.pop("ffa_decode_kernel", None)
    kwargs.pop("k_sample", None)
    kwargs.pop("k_full", None)

    # CUDA Graph 加速路径
    # 注意: CUDA Graph 不支持 return_skip_ratio 和 return_lse
    if cudagraph_runner is not None and not return_skip_ratio and not return_lse:
        # 检查是否已录制
        # 支持两种 runner:
        # 1. MultiLengthCudaGraphRunner: is_captured 是方法
        # 2. GlobalCudaGraphManager/CudaGraphRunnerWithPadding: is_captured 是属性
        if hasattr(cudagraph_runner, 'is_captured'):
            if callable(cudagraph_runner.is_captured):
                # MultiLengthCudaGraphRunner
                seq_len = k_q.shape[1]
                is_captured = cudagraph_runner.is_captured(seq_len)
            else:
                # GlobalCudaGraphManager or CudaGraphRunnerWithPadding
                is_captured = cudagraph_runner.is_captured
        else:
            is_captured = False

        if is_captured:
            # 重放 CUDA Graph
            return cudagraph_runner.replay(
                q=q,
                k_q=k_q,
                k_scale=k_scale,
                v=v,
                k_residual=k_residual,
                k_zero=None,  # 对称量化不需要 zero point
            )
        else:
            # 首次调用: 录制 CUDA Graph
            # 检查是否是 GlobalCudaGraphManager (需要传递 kernel_fn)
            if hasattr(cudagraph_runner, 'warmup') and 'kernel_fn' in cudagraph_runner.warmup.__code__.co_varnames:
                # GlobalCudaGraphManager
                return cudagraph_runner.warmup(
                    kernel_fn=attn_forward_decode_quantized,
                    q=q,
                    k_q=k_q,
                    k_scale=k_scale,
                    v=v,
                    k_residual=k_residual,
                    k_zero=None,
                    k_bits=k_bits,
                    scale=scale,
                    BS=BS,
                    SBS=SBS,
                    delta=delta,
                    use_fp8_residual=use_fp8_residual,
                    **kwargs,
                )
            else:
                # MultiLengthCudaGraphRunner
                return cudagraph_runner.warmup(
                    q=q,
                    k_q=k_q,
                    k_scale=k_scale,
                    v=v,
                    k_residual=k_residual,
                    k_zero=None,
                    k_bits=k_bits,
                    scale=scale,
                    BS=BS,
                    SBS=SBS,
                    delta=delta,
                    use_fp8_residual=use_fp8_residual,
                    **kwargs,
                )

    # 标准路径 (支持所有特性)
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

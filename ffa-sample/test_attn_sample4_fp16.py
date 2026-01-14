#!/usr/bin/env python3
"""
测试 attn_sample4_fp16.py kernel（采样 K 不量化，使用 FP16）

用法：
    python test_attn_sample4_fp16.py

也可以通过现有 benchmark 脚本测试：
    python run_attn_bench_sample4_q2.py --attn-kernel attn_sample4_fp16

对比 Q2 量化版本：
    python run_attn_bench_sample4_q2.py --attn-kernel attn_sample4_q2_sym
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# 确保可以导入 attn_kernel
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from attn_kernel.attn_sample4_fp16 import (
    attn_forward_decode_sample4,
    sample_k_fp16,
    quantize_k_sample4_2bit_symmetric,
    CUDAGraphDecodeRunnerSample4Q2,
)

# 也导入 Q2 版本用于对比
try:
    from attn_kernel.attn_sample4_q2_sym import (
        attn_forward_decode_sample4 as attn_forward_decode_q2,
        quantize_k_sample4_2bit_symmetric as quantize_k_q2,
    )
    HAS_Q2 = True
except ImportError:
    HAS_Q2 = False


def benchmark(fn, iters=100, warmup=20):
    """简单的 benchmark 函数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()

    return start.elapsed_time(end) / iters


def test_correctness(B=1, T=4096, HQ=32, HKV=8, K=128, V=128, BS=128, delta=5.0, dtype=torch.float16):
    """测试正确性：对比 FP16 采样和 Q2 量化采样的结果"""
    print(f"\n=== 正确性测试 ===")
    print(f"B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}, BS={BS}, delta={delta}")

    device = "cuda"
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # FP16 版本
    k_sample = sample_k_fp16(k, BS=BS)
    k_sample_scale_dummy = torch.zeros((B, k_sample.shape[1], HKV, K), device=device, dtype=dtype)

    out_fp16, skip_ratio_fp16 = attn_forward_decode_sample4(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale_dummy,
        k_full=k,
        v=v,
        BS=BS,
        delta=delta,
        return_skip_ratio=True,
    )

    print(f"FP16 版本 - Skip ratio: {skip_ratio_fp16:.4f}")
    print(f"FP16 输出 shape: {out_fp16.shape}, dtype: {out_fp16.dtype}")

    if HAS_Q2:
        # Q2 量化版本
        k_sample_q2, k_scale_q2 = quantize_k_q2(k, BS=BS)

        out_q2, skip_ratio_q2 = attn_forward_decode_q2(
            q=q,
            k_sample_q=k_sample_q2,
            k_sample_scale=k_scale_q2,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=True,
        )

        print(f"Q2 版本 - Skip ratio: {skip_ratio_q2:.4f}")

        # 对比输出
        diff = (out_fp16 - out_q2).abs()
        print(f"FP16 vs Q2 差异: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
    else:
        print("[跳过 Q2 对比，未找到 attn_sample4_q2_sym 模块]")

    print("正确性测试完成！")


def test_speed(B=1, T=65536, HQ=32, HKV=8, K=128, V=128, BS=128, delta=5.0, dtype=torch.float16, iters=100, warmup=20):
    """测试速度"""
    print(f"\n=== 速度测试 ===")
    print(f"B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}, BS={BS}, delta={delta}")
    print(f"iters={iters}, warmup={warmup}")

    device = "cuda"
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # FP16 版本
    k_sample = sample_k_fp16(k, BS=BS)
    k_sample_scale_dummy = torch.zeros((B, k_sample.shape[1], HKV, K), device=device, dtype=dtype)

    print(f"\nk_sample shape: {k_sample.shape}")
    print(f"k_sample 存储量: {k_sample.numel() * 2 / 1024 / 1024:.2f} MB")

    def run_fp16():
        return attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale_dummy,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=False,
        )

    # 获取 skip ratio
    _, skip_ratio_fp16 = attn_forward_decode_sample4(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale_dummy,
        k_full=k,
        v=v,
        BS=BS,
        delta=delta,
        return_skip_ratio=True,
    )

    ms_fp16 = benchmark(run_fp16, iters=iters, warmup=warmup)
    print(f"\nFP16 版本:")
    print(f"  延迟: {ms_fp16:.4f} ms")
    print(f"  Skip ratio: {skip_ratio_fp16:.4f}")

    # CUDAGraph 版本
    runner = CUDAGraphDecodeRunnerSample4Q2(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale_dummy,
        k_full=k,
        v=v,
        BS=BS,
        delta=delta,
        warmup=2,
    )

    ms_cg = benchmark(runner.replay_only, iters=iters, warmup=warmup)
    print(f"\nFP16 CUDAGraph 版本:")
    print(f"  延迟: {ms_cg:.4f} ms")
    print(f"  加速比: {ms_fp16 / ms_cg:.2f}x (vs 非 CUDAGraph)")

    if HAS_Q2:
        # Q2 量化版本
        k_sample_q2, k_scale_q2 = quantize_k_q2(k, BS=BS)

        print(f"\nk_sample_q2 shape: {k_sample_q2.shape}")
        print(f"k_sample_q2 存储量: {k_sample_q2.numel() / 1024 / 1024:.2f} MB")
        print(f"k_scale_q2 存储量: {k_scale_q2.numel() * 2 / 1024 / 1024:.2f} MB")

        def run_q2():
            return attn_forward_decode_q2(
                q=q,
                k_sample_q=k_sample_q2,
                k_sample_scale=k_scale_q2,
                k_full=k,
                v=v,
                BS=BS,
                delta=delta,
                return_skip_ratio=False,
            )

        _, skip_ratio_q2 = attn_forward_decode_q2(
            q=q,
            k_sample_q=k_sample_q2,
            k_sample_scale=k_scale_q2,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=True,
        )

        ms_q2 = benchmark(run_q2, iters=iters, warmup=warmup)
        print(f"\nQ2 版本:")
        print(f"  延迟: {ms_q2:.4f} ms")
        print(f"  Skip ratio: {skip_ratio_q2:.4f}")

        print(f"\n=== 对比总结 ===")
        print(f"FP16 vs Q2 速度比: {ms_q2 / ms_fp16:.2f}x")
        if ms_fp16 < ms_q2:
            print(f"FP16 更快 {(ms_q2 - ms_fp16) / ms_q2 * 100:.1f}%")
        else:
            print(f"Q2 更快 {(ms_fp16 - ms_q2) / ms_fp16 * 100:.1f}%")
    else:
        print("\n[跳过 Q2 对比，未找到 attn_sample4_q2_sym 模块]")

    print("\n速度测试完成！")


def main():
    parser = argparse.ArgumentParser(description="测试 attn_sample4_fp16 kernel")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--T", type=int, default=65536, help="Sequence length")
    parser.add_argument("--HQ", type=int, default=32, help="Query heads")
    parser.add_argument("--HKV", type=int, default=8, help="KV heads")
    parser.add_argument("--K", type=int, default=128, help="Head dim for K")
    parser.add_argument("--V", type=int, default=128, help="Head dim for V")
    parser.add_argument("--BS", type=int, default=128, help="Block size")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta threshold")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness test")
    parser.add_argument("--skip-speed", action="store_true", help="Skip speed test")
    args = parser.parse_args()

    print("=" * 60)
    print("attn_sample4_fp16 测试 (采样 K 不量化)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("错误：需要 CUDA 设备")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")

    if not args.skip_correctness:
        test_correctness(
            B=args.B,
            T=min(args.T, 8192),  # 正确性测试用较短序列
            HQ=args.HQ,
            HKV=args.HKV,
            K=args.K,
            V=args.V,
            BS=args.BS,
            delta=args.delta,
        )

    if not args.skip_speed:
        test_speed(
            B=args.B,
            T=args.T,
            HQ=args.HQ,
            HKV=args.HKV,
            K=args.K,
            V=args.V,
            BS=args.BS,
            delta=args.delta,
            iters=args.iters,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()

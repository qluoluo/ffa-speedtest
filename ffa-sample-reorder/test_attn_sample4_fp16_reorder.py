#!/usr/bin/env python3
"""
测试 attn_sample4_fp16_reorder.py kernel（采样点交换到序列前面）

用法：
    python test_attn_sample4_fp16_reorder.py

核心改进：
    - 原版：采样 K 额外存储为 [B, num_blocks, HKV, NUM_SAMPLES, K]
    - Reorder 版：将采样点交换到序列最前面，无需额外存储
    - 节省采样 K 的存储空间
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

from attn_kernel.attn_sample4_fp16_reorder import (
    attn_forward_decode_reorder,
    reorder_kv_for_sampling,
    get_sample_range,
    CUDAGraphDecodeRunnerReorder,
    SAMPLE_OFFSETS,
    NUM_SAMPLES,
)

# 导入原版用于对比
PARENT_DIR = THIS_DIR.parent / "ffa-sample"
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from attn_kernel.attn_sample4_fp16 import (
        attn_forward_decode_sample4 as attn_forward_original,
        sample_k_fp16,
    )
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print("[警告] 未找到原版 attn_sample4_fp16，跳过对比测试")


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


def test_reorder_function(B=1, T=1024, HKV=8, K=128, V=128, BS=128):
    """测试 reorder 函数的正确性"""
    print(f"\n=== Reorder 函数测试 ===")
    print(f"B={B}, T={T}, HKV={HKV}, K={K}, V={V}, BS={BS}")

    device = "cuda"
    dtype = torch.float16

    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # 执行 reorder
    k_reordered, v_reordered, reorder_indices, inverse_indices = reorder_kv_for_sampling(k, v, BS=BS)

    print(f"原始 k shape: {k.shape}")
    print(f"重排后 k shape: {k_reordered.shape}")
    print(f"reorder_indices shape: {reorder_indices.shape}")

    num_blocks = (T + BS - 1) // BS
    sample_range = get_sample_range(T, BS, NUM_SAMPLES)
    print(f"num_blocks: {num_blocks}")
    print(f"采样点范围: [0, {sample_range})")

    # 验证重排后的采样点位置
    print(f"\n采样点验证:")
    for block_idx in range(min(3, num_blocks)):  # 只检查前几个 block
        block_start = block_idx * BS
        for sample_idx, offset in enumerate(SAMPLE_OFFSETS):
            orig_pos = block_start + offset
            if orig_pos < T:
                reordered_pos = block_idx * NUM_SAMPLES + sample_idx
                # 检查 reorder_indices
                assert reorder_indices[reordered_pos].item() == orig_pos, \
                    f"Block {block_idx} sample {sample_idx}: expected {orig_pos}, got {reorder_indices[reordered_pos].item()}"
                # 检查值是否正确
                assert torch.allclose(k_reordered[:, reordered_pos], k[:, orig_pos]), \
                    f"Block {block_idx} sample {sample_idx}: value mismatch"
        print(f"  Block {block_idx}: OK")

    # 验证逆索引
    k_recovered = k_reordered[:, inverse_indices]
    assert torch.allclose(k_recovered, k), "逆重排失败！"
    print(f"\n逆重排验证: OK")

    print("Reorder 函数测试通过！")


def test_correctness(B=1, T=4096, HQ=32, HKV=8, K=128, V=128, BS=128, delta=5.0, dtype=torch.float16):
    """测试正确性：对比 reorder 版本和原版的结果"""
    print(f"\n=== 正确性测试 ===")
    print(f"B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}, BS={BS}, delta={delta}")

    device = "cuda"
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Reorder 版本
    k_reordered, v_reordered, _, _ = reorder_kv_for_sampling(k, v, BS=BS)

    out_reorder, skip_ratio_reorder = attn_forward_decode_reorder(
        q=q,
        k_reordered=k_reordered,
        v_reordered=v_reordered,
        BS=BS,
        delta=delta,
        return_skip_ratio=True,
    )

    print(f"Reorder 版本 - Skip ratio: {skip_ratio_reorder:.4f}")
    print(f"Reorder 输出 shape: {out_reorder.shape}, dtype: {out_reorder.dtype}")

    if HAS_ORIGINAL:
        # 原版
        k_sample = sample_k_fp16(k, BS=BS)
        k_sample_scale_dummy = torch.zeros((B, k_sample.shape[1], HKV, K), device=device, dtype=dtype)

        out_original, skip_ratio_original = attn_forward_original(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale_dummy,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=True,
        )

        print(f"原版 - Skip ratio: {skip_ratio_original:.4f}")

        # 对比输出
        diff = (out_reorder - out_original).abs()
        print(f"Reorder vs 原版差异: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

        # 由于 block 剪枝顺序可能不同，允许一定误差
        if diff.max().item() < 0.1:
            print("正确性验证: OK (误差在可接受范围)")
        else:
            print("正确性验证: 警告 - 输出差异较大，请检查")
    else:
        print("[跳过原版对比]")

    print("正确性测试完成！")


def test_memory_saving(B=1, T=65536, HKV=8, K=128, V=128, BS=128, dtype=torch.float16):
    """测试内存节省"""
    print(f"\n=== 内存对比 ===")
    print(f"B={B}, T={T}, HKV={HKV}, K={K}, V={V}, BS={BS}")

    device = "cuda"

    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    num_blocks = (T + BS - 1) // BS

    # 原版额外存储
    original_k_sample_size = B * num_blocks * HKV * NUM_SAMPLES * K * 2  # FP16
    original_k_sample_mb = original_k_sample_size / 1024 / 1024

    # Reorder 版本：无额外存储，只需要重排索引（可忽略）
    reorder_indices_size = T * 8  # int64
    reorder_indices_mb = reorder_indices_size / 1024 / 1024

    print(f"原版采样 K 额外存储: {original_k_sample_mb:.2f} MB")
    print(f"Reorder 版索引存储: {reorder_indices_mb:.4f} MB (可选，调试用)")
    print(f"节省: {original_k_sample_mb - reorder_indices_mb:.2f} MB")
    print(f"节省比例: {(1 - reorder_indices_mb / original_k_sample_mb) * 100:.1f}%")


def test_speed(B=1, T=65536, HQ=32, HKV=8, K=128, V=128, BS=128, delta=5.0, dtype=torch.float16, iters=100, warmup=20):
    """测试速度"""
    print(f"\n=== 速度测试 ===")
    print(f"B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}, BS={BS}, delta={delta}")
    print(f"iters={iters}, warmup={warmup}")

    device = "cuda"
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Reorder 版本
    k_reordered, v_reordered, _, _ = reorder_kv_for_sampling(k, v, BS=BS)

    def run_reorder():
        return attn_forward_decode_reorder(
            q=q,
            k_reordered=k_reordered,
            v_reordered=v_reordered,
            BS=BS,
            delta=delta,
            return_skip_ratio=False,
        )

    # 获取 skip ratio
    _, skip_ratio_reorder = attn_forward_decode_reorder(
        q=q,
        k_reordered=k_reordered,
        v_reordered=v_reordered,
        BS=BS,
        delta=delta,
        return_skip_ratio=True,
    )

    ms_reorder = benchmark(run_reorder, iters=iters, warmup=warmup)
    print(f"\nReorder 版本:")
    print(f"  延迟: {ms_reorder:.4f} ms")
    print(f"  Skip ratio: {skip_ratio_reorder:.4f}")

    # CUDAGraph 版本
    runner = CUDAGraphDecodeRunnerReorder(
        q=q,
        k_reordered=k_reordered,
        v_reordered=v_reordered,
        BS=BS,
        delta=delta,
        warmup=2,
    )

    ms_cg = benchmark(runner.replay_only, iters=iters, warmup=warmup)
    print(f"\nReorder CUDAGraph 版本:")
    print(f"  延迟: {ms_cg:.4f} ms")
    print(f"  加速比: {ms_reorder / ms_cg:.2f}x (vs 非 CUDAGraph)")

    if HAS_ORIGINAL:
        # 原版
        k_sample = sample_k_fp16(k, BS=BS)
        k_sample_scale_dummy = torch.zeros((B, k_sample.shape[1], HKV, K), device=device, dtype=dtype)

        def run_original():
            return attn_forward_original(
                q=q,
                k_sample_q=k_sample,
                k_sample_scale=k_sample_scale_dummy,
                k_full=k,
                v=v,
                BS=BS,
                delta=delta,
                return_skip_ratio=False,
            )

        _, skip_ratio_original = attn_forward_original(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale_dummy,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=True,
        )

        ms_original = benchmark(run_original, iters=iters, warmup=warmup)
        print(f"\n原版:")
        print(f"  延迟: {ms_original:.4f} ms")
        print(f"  Skip ratio: {skip_ratio_original:.4f}")

        print(f"\n=== 对比总结 ===")
        print(f"Reorder vs 原版速度比: {ms_original / ms_reorder:.2f}x")
        if ms_reorder < ms_original:
            print(f"Reorder 更快 {(ms_original - ms_reorder) / ms_original * 100:.1f}%")
        else:
            print(f"原版更快 {(ms_reorder - ms_original) / ms_reorder * 100:.1f}%")
    else:
        print("\n[跳过原版对比]")

    print("\n速度测试完成！")


def main():
    parser = argparse.ArgumentParser(description="测试 attn_sample4_fp16_reorder kernel")
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
    parser.add_argument("--skip-reorder-test", action="store_true", help="Skip reorder function test")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness test")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory test")
    parser.add_argument("--skip-speed", action="store_true", help="Skip speed test")
    args = parser.parse_args()

    print("=" * 60)
    print("attn_sample4_fp16_reorder 测试 (采样点交换到序列前面)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("错误：需要 CUDA 设备")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")

    if not args.skip_reorder_test:
        test_reorder_function(
            B=args.B,
            T=min(args.T, 2048),  # reorder 测试用较短序列
            HKV=args.HKV,
            K=args.K,
            V=args.V,
            BS=args.BS,
        )

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

    if not args.skip_memory:
        test_memory_saving(
            B=args.B,
            T=args.T,
            HKV=args.HKV,
            K=args.K,
            V=args.V,
            BS=args.BS,
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

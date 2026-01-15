"""
Basic usage example for FFA-Sample.

展示如何使用 FFA-Sample 进行稀疏注意力计算。
"""

import torch
import time

# 添加项目路径
import sys
sys.path.insert(0, "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-flashinfer-sample")

from ffa_sample import (
    SparseAttentionWithFlashInfer,
    sample_k_fp16,
    attn_forward_decode_sample4,
)
from ffa_sample.utils import continuous_to_paged


def example_basic_usage():
    """基本使用示例."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # 配置
    B = 1  # Batch size
    T = 4096  # Sequence length
    HQ = 32  # Query heads
    HKV = 8  # KV heads (GQA)
    K = 128  # Head dimension
    V = 128  # Value dimension
    BS = 128  # Block size

    device = "cuda:0"
    dtype = torch.float16

    print(f"Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  Query heads: {HQ}, KV heads: {HKV}")
    print(f"  Head dimension: {K}")
    print(f"  Block size: {BS}")
    print()

    # 创建输入
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # 方法 1: 使用类接口
    print("Method 1: Using SparseAttentionWithFlashInfer class")
    sparse_attn = SparseAttentionWithFlashInfer(
        num_heads=HQ,
        head_dim=K,
        page_size=BS,
        num_kv_heads=HKV,
        device=device,
        dtype=dtype,
    )

    # Warmup
    for _ in range(3):
        _ = sparse_attn(q, k, v, delta=5.0)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        output = sparse_attn(q, k, v, delta=5.0)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  Output shape: {output.shape}")
    print(f"  Time per iteration: {elapsed / num_iters * 1000:.3f} ms")
    print()

    # 方法 2: 使用函数接口
    print("Method 2: Using functional interface")

    # 预处理: 提取采样 K
    k_sample = sample_k_fp16(k, BS=BS)
    print(f"  k_sample shape: {k_sample.shape}")
    print(f"  (B, num_blocks, HKV, NUM_SAMPLES, K)")

    # 创建 dummy scale
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

    # 执行稀疏注意力
    output, skip_ratio = attn_forward_decode_sample4(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale,
        k_full=k,
        v=v,
        BS=BS,
        delta=5.0,
        return_skip_ratio=True,
    )

    print(f"  Output shape: {output.shape}")
    print(f"  Skip ratio: {skip_ratio:.2%}")
    print()


def example_compare_with_full_attention():
    """与全注意力对比."""
    print("=" * 60)
    print("Example 2: Compare with Full Attention")
    print("=" * 60)

    B, T, HQ, K, V = 1, 2048, 32, 128, 128
    HKV = 8
    BS = 128
    device = "cuda:0"
    dtype = torch.float16

    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Full attention (PyTorch)
    print("Computing full attention...")
    q_expanded = q.squeeze(1)  # [B, HQ, K]

    # 扩展 K, V 以匹配 Q 头数 (GQA)
    G = HQ // HKV
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, K)
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, V)

    # [B, HQ, 1, K] @ [B, HQ, K, T] -> [B, HQ, 1, T]
    scale = 1.0 / (K ** 0.5)
    scores = torch.einsum("bhk,bthk->bht", q_expanded, k_expanded) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    # [B, HQ, T] @ [B, T, HQ, V] -> [B, HQ, V]
    full_output = torch.einsum("bht,bthv->bhv", attn_weights, v_expanded)

    print(f"  Full attention output shape: {full_output.shape}")

    # Sparse attention
    print("Computing sparse attention...")
    k_sample = sample_k_fp16(k, BS=BS)
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

    sparse_output, skip_ratio = attn_forward_decode_sample4(
        q=q,
        k_sample_q=k_sample,
        k_sample_scale=k_sample_scale,
        k_full=k,
        v=v,
        BS=BS,
        delta=5.0,
        return_skip_ratio=True,
    )

    print(f"  Sparse attention output shape: {sparse_output.shape}")
    print(f"  Skip ratio: {skip_ratio:.2%}")

    # 比较输出
    # Note: 由于稀疏注意力跳过了一些 block，输出会有差异
    diff = (sparse_output - full_output).abs()
    print(f"  Max absolute difference: {diff.max().item():.6f}")
    print(f"  Mean absolute difference: {diff.mean().item():.6f}")
    print()


def example_different_deltas():
    """测试不同的 delta 值."""
    print("=" * 60)
    print("Example 3: Effect of Delta Parameter")
    print("=" * 60)

    B, T, HQ, K, V = 1, 4096, 32, 128, 128
    HKV = 8
    BS = 128
    device = "cuda:0"
    dtype = torch.float16

    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    k_sample = sample_k_fp16(k, BS=BS)
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

    print(f"Testing different delta values (sequence length={T}):")
    print()

    for delta in [1.0, 3.0, 5.0, 7.0, 10.0]:
        _, skip_ratio = attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            BS=BS,
            delta=delta,
            return_skip_ratio=True,
        )
        kept_ratio = 1.0 - skip_ratio
        num_kept_blocks = int(num_blocks * kept_ratio)
        print(f"  delta={delta:5.1f}: skip={skip_ratio:6.2%}, kept blocks={num_kept_blocks}/{num_blocks}")

    print()


def example_benchmark():
    """性能基准测试."""
    print("=" * 60)
    print("Example 4: Performance Benchmark")
    print("=" * 60)

    device = "cuda:0"
    dtype = torch.float16
    HQ, HKV, K, V = 32, 8, 128, 128
    BS = 128

    seq_lengths = [1024, 2048, 4096, 8192, 16384]

    print("Sequence Length | Sparse (ms) | Skip Ratio")
    print("-" * 45)

    for T in seq_lengths:
        q = torch.randn(1, 1, HQ, K, device=device, dtype=dtype)
        k = torch.randn(1, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(1, T, HKV, V, device=device, dtype=dtype)

        k_sample = sample_k_fp16(k, BS=BS)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros((1, num_blocks, HKV, K), device=device, dtype=dtype)

        # Warmup
        for _ in range(5):
            _ = attn_forward_decode_sample4(
                q=q,
                k_sample_q=k_sample,
                k_sample_scale=k_sample_scale,
                k_full=k,
                v=v,
                BS=BS,
                delta=5.0,
            )
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 50
        start = time.time()
        for _ in range(num_iters):
            output, skip_ratio = attn_forward_decode_sample4(
                q=q,
                k_sample_q=k_sample,
                k_sample_scale=k_sample_scale,
                k_full=k,
                v=v,
                BS=BS,
                delta=5.0,
                return_skip_ratio=True,
            )
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters * 1000

        print(f"{T:>15} | {elapsed:>11.3f} | {skip_ratio:>10.2%}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FFA-Sample: Sparse Attention Examples")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_compare_with_full_attention()
    example_different_deltas()
    example_benchmark()

    print("All examples completed!")

"""
完整的 FFA-Sample 性能分析脚本

测试内容:
1. 不同序列长度的性能对比
2. 与 PyTorch 全注意力和 FlashAttention 的对比
3. 稀疏率对性能的影响
4. 准确性验证
"""

import sys
import time
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-flashinfer-sample")

from ffa_sample.kernels import sample_k_fp16, attn_forward_decode_sample4


def create_attention_pattern_data(
    B: int, T: int, HQ: int, HKV: int, K: int, V: int,
    pattern: str = "local",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """
    创建具有特定注意力模式的数据.

    pattern:
    - "random": 随机数据
    - "local": 局部注意力模式 (最近的位置权重更高)
    - "sparse": 稀疏注意力模式 (只有少数位置重要)
    """
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)

    if pattern == "random":
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)
    elif pattern == "local":
        # 让最近的 K 与 Q 更相似
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        # 增强最后几个位置的相似性
        local_window = min(256, T)
        k[:, -local_window:] = q.squeeze(1).unsqueeze(1).expand(-1, local_window, -1, -1)[:, :, :HKV, :] + 0.1 * torch.randn(B, local_window, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)
    elif pattern == "sparse":
        # 只有少数位置的 K 与 Q 相关
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype) * 0.1  # 大部分位置噪声小
        # 随机选择一些位置作为重要位置
        num_important = max(T // 10, 10)
        important_indices = torch.randperm(T)[:num_important].sort()[0]
        for idx in important_indices:
            k[:, idx] = q.squeeze(1)[:, :HKV, :] + 0.1 * torch.randn(B, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return q, k, v


def compute_full_attention(q, k, v):
    """计算全注意力作为参考."""
    B, Tq, HQ, K = q.shape
    _, T, HKV, _ = k.shape
    V = v.shape[-1]
    G = HQ // HKV
    scale = 1.0 / math.sqrt(K)

    k_exp = k.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, K)
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, V)
    q_2d = q.squeeze(1)

    scores = torch.einsum("bhk,bthk->bht", q_2d, k_exp) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bht,bthv->bhv", attn_weights, v_exp)

    return output


def test_accuracy(seq_len=4096, delta=5.0, page_size=128, max_kept_ratio=0.3):
    """测试稀疏注意力的准确性."""
    print("=" * 70)
    print("Accuracy Test")
    print("=" * 70)

    HQ, HKV, K, V = 32, 8, 128, 128
    device = "cuda:0"
    dtype = torch.float16

    for pattern in ["random", "local", "sparse"]:
        print(f"\nPattern: {pattern}")

        q, k, v = create_attention_pattern_data(1, seq_len, HQ, HKV, K, V, pattern, device, dtype)

        # Full attention
        full_output = compute_full_attention(q, k, v)

        # Sparse attention
        k_sample = sample_k_fp16(k, BS=page_size)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros((1, num_blocks, HKV, K), device=device, dtype=dtype)

        sparse_output, skip_ratio = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=page_size, delta=delta,
            max_kept_ratio=max_kept_ratio, return_skip_ratio=True,
        )

        # Compare
        diff = (sparse_output - full_output).abs()
        relative_err = diff / (full_output.abs() + 1e-6)

        print(f"  Skip ratio: {skip_ratio:.1%}")
        print(f"  Max absolute error: {diff.max().item():.6f}")
        print(f"  Mean absolute error: {diff.mean().item():.6f}")
        print(f"  Max relative error: {relative_err.max().item():.2%}")
        print(f"  Mean relative error: {relative_err.mean().item():.2%}")


def benchmark_with_patterns(seq_lengths=[4096, 8192, 16384, 32768]):
    """使用不同注意力模式进行基准测试."""
    print("\n" + "=" * 80)
    print("Benchmark with Different Attention Patterns")
    print("=" * 80)

    HQ, HKV, K, V = 32, 8, 128, 128
    page_size = 128
    delta = 5.0
    device = "cuda:0"
    dtype = torch.float16
    num_warmup, num_iters = 5, 30

    for pattern in ["random", "local", "sparse"]:
        print(f"\n{'='*30} Pattern: {pattern} {'='*30}")
        print(f"{'SeqLen':>8} | {'PyTorch(ms)':>11} | {'FFA(ms)':>10} | {'Skip':>8} | {'Speedup':>10}")
        print("-" * 60)

        for seq_len in seq_lengths:
            q, k, v = create_attention_pattern_data(1, seq_len, HQ, HKV, K, V, pattern, device, dtype)

            # PyTorch baseline
            for _ in range(num_warmup):
                _ = compute_full_attention(q, k, v)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(num_iters):
                _ = compute_full_attention(q, k, v)
            torch.cuda.synchronize()
            pytorch_ms = (time.perf_counter() - start) / num_iters * 1000

            # FFA-Sample
            k_sample = sample_k_fp16(k, BS=page_size)
            num_blocks = k_sample.shape[1]
            k_sample_scale = torch.zeros((1, num_blocks, HKV, K), device=device, dtype=dtype)

            for _ in range(num_warmup):
                _ = attn_forward_decode_sample4(
                    q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
                    k_full=k, v=v, BS=page_size, delta=delta,
                )
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(num_iters):
                output, skip_ratio = attn_forward_decode_sample4(
                    q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
                    k_full=k, v=v, BS=page_size, delta=delta,
                    return_skip_ratio=True,
                )
            torch.cuda.synchronize()
            ffa_ms = (time.perf_counter() - start) / num_iters * 1000

            speedup = pytorch_ms / ffa_ms
            print(f"{seq_len:>8} | {pytorch_ms:>11.3f} | {ffa_ms:>10.3f} | {skip_ratio:>8.1%} | {speedup:>10.2f}x")

            del q, k, v, k_sample
            torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 80)
    print("FFA-Sample Comprehensive Analysis")
    print("=" * 80)

    # 1. 准确性测试
    test_accuracy()

    # 2. 不同注意力模式的基准测试
    benchmark_with_patterns()

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

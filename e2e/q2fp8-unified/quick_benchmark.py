"""
快速性能对比：融合 vs 原始实现

对比融合 RoPE + 量化前后的性能
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ffa_model"))

from q2fp8_cache import Q2FP8SymCache

def benchmark_cache_update(use_fused: bool, seq_len: int = 8192, num_runs: int = 10):
    """
    Benchmark cache update with/without fused RoPE
    """
    device = torch.device('cuda')
    B, HKV, K, V = 1, 8, 128, 128
    BS = 128
    k_bits = 2

    # 创建 cache
    cache = Q2FP8SymCache(BS=BS, k_bits=k_bits)

    # 生成测试数据
    torch.manual_seed(42)
    key_states = torch.randn(B, seq_len, HKV, K, dtype=torch.float16, device=device)
    value_states = torch.randn(B, seq_len, HKV, V, dtype=torch.float16, device=device)

    if use_fused:
        # 使用融合实现（提供 cos/sin）
        cos = torch.randn(B, seq_len, K, dtype=torch.float16, device=device)
        sin = torch.randn(B, seq_len, K, dtype=torch.float16, device=device)
        cache_kwargs = {"cos": cos, "sin": sin}
    else:
        # 不使用融合（不提供 cos/sin）
        cache_kwargs = None

    # Warmup
    for _ in range(3):
        cache.reset()
        cache.update(key_states, value_states, layer_idx=0, cache_kwargs=cache_kwargs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        cache.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        cache.update(key_states, value_states, layer_idx=0, cache_kwargs=cache_kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return times

def main():
    print("="*70)
    print("Performance Comparison: Fused vs Original")
    print("="*70)

    for seq_len in [1024, 4096, 8192]:
        print(f"\nSequence length: {seq_len}")
        print("-" * 70)

        # 测试原始实现（不带融合）
        print("Testing original (without fused RoPE)...")
        times_original = benchmark_cache_update(use_fused=False, seq_len=seq_len)
        avg_original = sum(times_original) / len(times_original)

        # 测试融合实现
        print("Testing fused (with fused RoPE + quantization)...")
        times_fused = benchmark_cache_update(use_fused=True, seq_len=seq_len)
        avg_fused = sum(times_fused) / len(times_fused)

        # 结果
        speedup = avg_original / avg_fused
        saved = avg_original - avg_fused

        print(f"\nResults:")
        print(f"  Original:  {avg_original:.3f} ms")
        print(f"  Fused:     {avg_fused:.3f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Time saved: {saved:.3f} ms ({100*saved/avg_original:.1f}%)")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()

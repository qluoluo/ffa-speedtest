"""
详细的性能分析：找出瓶颈
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ffa_model"))

from q2fp8_cache import Q2FP8SymCache

def detailed_benchmark():
    device = torch.device('cuda')
    B, seq_len, HKV, K, V = 1, 8192, 8, 128, 128
    BS = 128
    k_bits = 2

    # 创建 cache
    cache_no_fused = Q2FP8SymCache(BS=BS, k_bits=k_bits)
    cache_fused = Q2FP8SymCache(BS=BS, k_bits=k_bits)

    # 生成测试数据
    torch.manual_seed(42)
    key_states = torch.randn(B, seq_len, HKV, K, dtype=torch.float16, device=device)
    value_states = torch.randn(B, seq_len, HKV, V, dtype=torch.float16, device=device)
    cos = torch.randn(B, seq_len, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, seq_len, K, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(3):
        cache_no_fused.reset()
        cache_no_fused.update(key_states, value_states, layer_idx=0, cache_kwargs=None)
        cache_fused.reset()
        cache_fused.update(key_states, value_states, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})
    torch.cuda.synchronize()

    # Test without fused
    print("Testing without fused RoPE...")
    times_no_fused = []
    for _ in range(10):
        cache_no_fused.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        cache_no_fused.update(key_states, value_states, layer_idx=0, cache_kwargs=None)
        torch.cuda.synchronize()
        times_no_fused.append((time.perf_counter() - start) * 1000)

    # Test with fused
    print("Testing with fused RoPE...")
    times_fused = []
    for _ in range(10):
        cache_fused.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        cache_fused.update(key_states, value_states, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})
        torch.cuda.synchronize()
        times_fused.append((time.perf_counter() - start) * 1000)

    avg_no_fused = sum(times_no_fused) / len(times_no_fused)
    avg_fused = sum(times_fused) / len(times_fused)

    print(f"\nResults for {seq_len} tokens:")
    print(f"  Without fused: {avg_no_fused:.3f} ms")
    print(f"  With fused:    {avg_fused:.3f} ms")
    print(f"  Difference:    {avg_fused - avg_no_fused:.3f} ms")
    print(f"  Speedup:       {avg_no_fused / avg_fused:.2f}x")

    # 分析 cos/sin 提取的开销
    print("\nAnalyzing cos/sin extraction overhead...")
    times_extract = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        # 模拟提取操作
        start_pos = 0
        full_blocks_len = (seq_len // BS) * BS
        cos_blocks = cos[:, start_pos:start_pos + full_blocks_len, :]
        sin_blocks = sin[:, start_pos:start_pos + full_blocks_len, :]
        torch.cuda.synchronize()
        times_extract.append((time.perf_counter() - start) * 1000)

    print(f"  Cos/sin extraction: {sum(times_extract)/len(times_extract):.3f} ms")

if __name__ == "__main__":
    detailed_benchmark()

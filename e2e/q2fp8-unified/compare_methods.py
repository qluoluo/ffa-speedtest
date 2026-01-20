"""
对比测试：找出真正的性能差异
"""

import torch
import time

device = torch.device('cuda')
B, T, HKV, K = 1, 8192, 8, 128
BS = 128
k_bits = 2

# 生成测试数据
torch.manual_seed(42)
k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

# 导入函数
import sys
sys.path.insert(0, 'ffa_model')
from fused_rope_quant import fused_rope_and_quantize
from q2fp8_cache import quantize_symmetric_blocks

# Warmup
for _ in range(5):
    _ = quantize_symmetric_blocks(k, BS, k_bits)
    _ = fused_rope_and_quantize(k, cos, sin, BS, k_bits)
torch.cuda.synchronize()

print("="*70)
print("Direct function call comparison")
print("="*70)

# Test 1: 只量化
times = []
for _ in range(20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = quantize_symmetric_blocks(k, BS, k_bits)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
print(f"Quantize only:        {sum(times)/len(times):.3f} ms")

# Test 2: 融合 RoPE + 量化
times = []
for _ in range(20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = fused_rope_and_quantize(k, cos, sin, BS, k_bits)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
print(f"Fused RoPE+Quant:     {sum(times)/len(times):.3f} ms")

# Test 3: 分离的 RoPE + 量化
def separate_rope_and_quant(k, cos, sin, BS, k_bits):
    # Apply RoPE
    cos_exp = cos.unsqueeze(2)
    sin_exp = sin.unsqueeze(2)
    k1 = k[..., :K//2]
    k2 = k[..., K//2:]
    k_rotated = torch.cat([
        k1 * cos_exp[..., :K//2] - k2 * sin_exp[..., :K//2],
        k2 * cos_exp[..., K//2:] + k1 * sin_exp[..., K//2:]
    ], dim=-1)
    # Quantize
    return quantize_symmetric_blocks(k_rotated, BS, k_bits)

times = []
for _ in range(20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = separate_rope_and_quant(k, cos, sin, BS, k_bits)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
print(f"Separate RoPE+Quant:  {sum(times)/len(times):.3f} ms")

print("\n" + "="*70)
print("Conclusion:")
print("="*70)
print("If fused is faster than separate, the fusion is working!")
print("If fused is slower than quantize-only, there's overhead in the fusion.")

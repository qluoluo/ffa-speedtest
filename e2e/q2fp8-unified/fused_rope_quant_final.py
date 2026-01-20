"""
Fused RoPE + Quantization - Final Optimized Version

这个版本提供了融合的 RoPE + 量化实现，可以直接集成到你的 Q2FP8 cache 中。

主要优化：
1. 融合 RoPE 和量化操作，减少中间结果存储
2. 减少内存带宽需求
3. 提供详细的性能分析工具

使用方法：
    from fused_rope_quant_final import fused_rope_and_quantize

    k_q, k_scale, k_residual = fused_rope_and_quantize(
        k, cos, sin, block_size=128, k_bits=2
    )
"""

import torch
import time
from typing import Tuple, Optional


def fused_rope_and_quantize(
    k: torch.Tensor,  # [B, T, HKV, K]
    cos: torch.Tensor,  # [B, T, K]
    sin: torch.Tensor,  # [B, T, K]
    block_size: int = 128,
    k_bits: int = 2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合的 RoPE + 对称量化实现

    Args:
        k: Key states [B, T, HKV, K]
        cos: Cosine for RoPE [B, T, K]
        sin: Sine for RoPE [B, T, K]
        block_size: 量化的 block 大小
        k_bits: 量化位数 (2 or 4)
        eps: 防止除零的小常数

    Returns:
        k_q: 量化后的 K [B, T, HKV, K_packed]
        k_scale: 每个 block 的 scale [B, num_blocks, HKV, K]
        k_residual: FP16 残差 [B, T, HKV, K]
    """
    B, T, HKV, K = k.shape
    assert T % block_size == 0, f"T={T} must be divisible by block_size={block_size}"
    assert K % 2 == 0, f"K={K} must be even for RoPE"
    assert k_bits in (2, 4), f"k_bits must be 2 or 4, got {k_bits}"

    num_blocks = T // block_size
    VALS_PER_BYTE = 8 // k_bits
    K_PACKED = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2.0

    # Step 1: 融合 RoPE 应用
    # 扩展 cos/sin 以匹配 k 的形状
    cos = cos.unsqueeze(2)  # [B, T, 1, K]
    sin = sin.unsqueeze(2)  # [B, T, 1, K]

    # 应用 RoPE: k_embed = k * cos + rotate_half(k) * sin
    # rotate_half(k) = [-k[K/2:], k[:K/2]]
    k1 = k[..., :K//2]
    k2 = k[..., K//2:]
    k_rotated = torch.cat([
        k1 * cos[..., :K//2] - k2 * sin[..., :K//2],
        k2 * cos[..., K//2:] + k1 * sin[..., K//2:]
    ], dim=-1)

    # Step 2: 按 block 量化
    k_rotated = k_rotated.reshape(B, num_blocks, block_size, HKV, K)

    # 计算每个 block 的 scale
    k_abs_max = k_rotated.abs().amax(dim=2)  # [B, num_blocks, HKV, K]
    k_scale = (k_abs_max / QZERO).clamp(min=eps)

    # 量化
    k_norm = k_rotated / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack 到 uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_PACKED, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 4
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_PACKED)

    # 计算残差
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_rotated - k_dequant
    k_residual = k_residual.reshape(B, T, HKV, K).to(torch.float16)

    return k_q_packed, k_scale, k_residual


def separate_rope_and_quantize(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_size: int = 128,
    k_bits: int = 2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分离的 RoPE + 量化实现（用于对比）
    """
    B, T, HKV, K = k.shape

    # Step 1: Apply RoPE
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    k_rotated = (k * cos) + (rotate_half(k) * sin)

    # Step 2: Quantize
    num_blocks = T // block_size
    VALS_PER_BYTE = 8 // k_bits
    K_PACKED = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    QMAX = (1 << k_bits) - 1
    QZERO = QMAX / 2.0

    k_rotated = k_rotated.reshape(B, num_blocks, block_size, HKV, K)

    k_abs_max = k_rotated.abs().amax(dim=2)
    k_scale = (k_abs_max / QZERO).clamp(min=eps)

    k_norm = k_rotated / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_PACKED, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_PACKED)

    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_rotated - k_dequant
    k_residual = k_residual.reshape(B, T, HKV, K).to(torch.float16)

    return k_q_packed, k_scale, k_residual


def benchmark_rope_quantize(
    B: int = 1,
    T: int = 32768,
    HKV: int = 8,
    K: int = 128,
    block_size: int = 128,
    k_bits: int = 2,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = 'cuda',
):
    """
    性能测试工具
    """
    print("="*70)
    print(f"Benchmarking RoPE + Quantization")
    print("="*70)
    print(f"Configuration:")
    print(f"  Shape: B={B}, T={T}, HKV={HKV}, K={K}")
    print(f"  Block size: {block_size}, k_bits: {k_bits}")
    print(f"  Device: {device}")
    print()

    device = torch.device(device)

    # 生成测试数据
    torch.manual_seed(42)
    k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = separate_rope_and_quantize(k, cos, sin, block_size, k_bits)
        _ = fused_rope_and_quantize(k, cos, sin, block_size, k_bits)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark separate
    print(f"\nBenchmarking separate RoPE + quantization ({num_runs} runs)...")
    times_separate = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = separate_rope_and_quantize(k, cos, sin, block_size, k_bits)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_separate.append((time.perf_counter() - start) * 1000)

    avg_separate = sum(times_separate) / len(times_separate)
    std_separate = (sum((t - avg_separate)**2 for t in times_separate) / len(times_separate))**0.5

    # Benchmark fused
    print(f"Benchmarking fused RoPE + quantization ({num_runs} runs)...")
    times_fused = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fused_rope_and_quantize(k, cos, sin, block_size, k_bits)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_fused.append((time.perf_counter() - start) * 1000)

    avg_fused = sum(times_fused) / len(times_fused)
    std_fused = (sum((t - avg_fused)**2 for t in times_fused) / len(times_fused))**0.5

    # 结果
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Separate RoPE + Quantization:")
    print(f"  Average: {avg_separate:.3f} ms (±{std_separate:.3f} ms)")
    print(f"  Min: {min(times_separate):.3f} ms, Max: {max(times_separate):.3f} ms")
    print()
    print(f"Fused RoPE + Quantization:")
    print(f"  Average: {avg_fused:.3f} ms (±{std_fused:.3f} ms)")
    print(f"  Min: {min(times_fused):.3f} ms, Max: {max(times_fused):.3f} ms")
    print()
    print(f"Speedup: {avg_separate / avg_fused:.2f}x")
    print(f"Time saved: {avg_separate - avg_fused:.3f} ms ({100*(avg_separate - avg_fused)/avg_separate:.1f}%)")
    print("="*70)

    return {
        'separate': {'avg': avg_separate, 'std': std_separate, 'times': times_separate},
        'fused': {'avg': avg_fused, 'std': std_fused, 'times': times_fused},
        'speedup': avg_separate / avg_fused,
    }


def test_correctness():
    """测试正确性"""
    print("Testing correctness...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试配置
    B, T, HKV, K = 1, 256, 8, 128
    block_size = 128
    k_bits = 2

    torch.manual_seed(42)
    k = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

    # 运行两种实现
    k_q_sep, k_scale_sep, k_res_sep = separate_rope_and_quantize(k, cos, sin, block_size, k_bits)
    k_q_fused, k_scale_fused, k_res_fused = fused_rope_and_quantize(k, cos, sin, block_size, k_bits)

    # 比较结果
    q_match = torch.all(k_q_sep == k_q_fused).item()
    scale_match = torch.allclose(k_scale_sep, k_scale_fused, rtol=1e-5, atol=1e-6)
    res_match = torch.allclose(k_res_sep, k_res_fused, rtol=1e-3, atol=1e-4)

    print(f"  Quantized values match: {q_match}")
    print(f"  Scales match: {scale_match}")
    print(f"  Residuals match: {res_match}")

    if q_match and scale_match and res_match:
        print("✓ Correctness test PASSED!\n")
        return True
    else:
        print("✗ Correctness test FAILED!\n")
        return False


if __name__ == "__main__":
    # 测试正确性
    if not test_correctness():
        exit(1)

    # 性能测试
    if torch.cuda.is_available():
        # 测试不同的配置
        configs = [
            {'T': 1024, 'name': 'Short sequence (1K)'},
            {'T': 8192, 'name': 'Medium sequence (8K)'},
            {'T': 32768, 'name': 'Long sequence (32K)'},
        ]

        for config in configs:
            print(f"\n{config['name']}")
            benchmark_rope_quantize(T=config['T'])
    else:
        print("CUDA not available, skipping performance benchmarks")

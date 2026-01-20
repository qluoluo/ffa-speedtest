#!/usr/bin/env python
"""
简单测试：验证 Q2FP8SymCache 的量化/反量化是否正确
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')

import torch
from q2fp8_cache import Q2FP8SymCache, quantize_symmetric

def test_quantization():
    print("=" * 60)
    print("Testing Q2FP8SymCache quantization/dequantization")
    print("=" * 60)

    # 创建测试数据
    B, T, HKV, K = 1, 256, 8, 128  # 2 blocks (BS=128)
    torch.manual_seed(42)
    k_orig = torch.randn(B, T, HKV, K, dtype=torch.float16, device='cuda')
    v_orig = torch.randn(B, T, HKV, K, dtype=torch.float16, device='cuda')

    print(f"Original K shape: {k_orig.shape}")
    print(f"Original K range: [{k_orig.min():.4f}, {k_orig.max():.4f}]")

    # 创建 cache 并分批更新
    cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)

    # 模拟逐 token 添加
    for i in range(T):
        k_new = k_orig[:, i:i+1, :, :]  # [B, 1, HKV, K]
        v_new = v_orig[:, i:i+1, :, :]
        keys, values = cache.update(k_new, v_new, layer_idx=0)

    print(f"\nAfter {T} tokens:")
    layer = cache.layers[0]
    print(f"  Quantized blocks: {layer.num_full_blocks}")
    print(f"  Quantized tokens: {layer.get_quantized_len()}")
    print(f"  Current tokens: {layer.get_current_len()}")
    print(f"  Total seq length: {layer.get_seq_length()}")

    # 获取反量化的 key
    k_reconstructed = layer.keys
    print(f"\nReconstructed K shape: {k_reconstructed.shape}")
    print(f"Reconstructed K range: [{k_reconstructed.min():.4f}, {k_reconstructed.max():.4f}]")

    # 计算误差
    error = (k_orig - k_reconstructed).abs()
    print(f"\n=== Reconstruction Error ===")
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
    print(f"Relative error: {(error / (k_orig.abs() + 1e-8)).mean():.4%}")

    # 分别检查量化部分和未量化部分
    quantized_len = layer.get_quantized_len()
    if quantized_len > 0:
        error_q = (k_orig[:, :quantized_len] - k_reconstructed[:, :quantized_len]).abs()
        print(f"\nQuantized part (first {quantized_len} tokens):")
        print(f"  Mean error: {error_q.mean():.6f}")
        print(f"  Max error: {error_q.max():.6f}")

    current_len = layer.get_current_len()
    if current_len > 0:
        error_c = (k_orig[:, quantized_len:] - k_reconstructed[:, quantized_len:]).abs()
        print(f"\nUnquantized part (last {current_len} tokens):")
        print(f"  Mean error: {error_c.mean():.6f}")
        print(f"  Max error: {error_c.max():.6f}")

    # 检查 k_scale 形状
    print(f"\n=== Scale Info ===")
    print(f"k_scale shape: {layer.k_scale.shape}")
    print(f"k_scale range: [{layer.k_scale.min():.4f}, {layer.k_scale.max():.4f}]")

    # 验证量化/反量化的一致性
    print("\n=== Direct quantize/dequantize test ===")
    k_test = k_orig[:, :128, :, :]  # First block
    k_q, k_scale, k_residual = quantize_symmetric(k_test, k_bits=2)

    # 手动反量化
    QMAX, QZERO = 3, 1.5
    VALS_PER_BYTE = 4
    K_packed = k_q.shape[-1]

    # Unpack
    k_unpacked = torch.stack([
        (k_q >> 0) & 0x3,
        (k_q >> 2) & 0x3,
        (k_q >> 4) & 0x3,
        (k_q >> 6) & 0x3,
    ], dim=-1).view(B, 128, HKV, -1)[..., :K].float()

    k_dequant = (k_unpacked - QZERO) * k_scale.unsqueeze(1)
    if k_residual is not None:
        k_dequant = k_dequant + k_residual.to(k_dequant.dtype)

    k_dequant = k_dequant.to(torch.float16)

    error_direct = (k_test - k_dequant).abs()
    print(f"Direct quantize/dequantize error:")
    print(f"  Mean error: {error_direct.mean():.6f}")
    print(f"  Max error: {error_direct.max():.6f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_quantization()

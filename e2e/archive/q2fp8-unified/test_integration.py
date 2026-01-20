"""
测试融合 RoPE + 量化的集成

验证修改后的代码是否正确工作
"""

import torch
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/e2e/q2fp8-unified/ffa_model')

from q2fp8_cache import Q2FP8SymCache

def test_fused_integration():
    print("Testing fused RoPE + quantization integration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 配置
    B, T, HKV, K, V = 1, 256, 8, 128, 128
    BS = 128
    k_bits = 2

    # 创建 cache
    cache = Q2FP8SymCache(BS=BS, k_bits=k_bits)

    # 生成测试数据
    torch.manual_seed(42)
    key_states = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)
    value_states = torch.randn(B, T, HKV, V, dtype=torch.float16, device=device)

    # 生成 cos/sin
    cos = torch.randn(B, T, K, dtype=torch.float16, device=device)
    sin = torch.randn(B, T, K, dtype=torch.float16, device=device)

    cache_kwargs = {
        "cos": cos,
        "sin": sin,
        "cache_position": torch.arange(T, device=device),
    }

    print(f"\nInput shapes:")
    print(f"  key_states: {key_states.shape}")
    print(f"  value_states: {value_states.shape}")
    print(f"  cos: {cos.shape}")
    print(f"  sin: {sin.shape}")

    # 测试 update
    print("\nTesting cache.update()...")
    try:
        key_out, value_out = cache.update(
            key_states, value_states, layer_idx=0, cache_kwargs=cache_kwargs
        )
        print(f"✓ Update successful!")
        print(f"  Output key shape: {key_out.shape}")
        print(f"  Output value shape: {value_out.shape}")
        print(f"  Cache seq length: {cache.get_seq_length()}")
        print(f"  Quantized length: {cache.get_quantized_len()}")
        print(f"  Current length: {cache.get_current_len()}")
    except Exception as e:
        print(f"✗ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 decode (单个 token)
    print("\nTesting decode (single token)...")
    try:
        key_new = torch.randn(B, 1, HKV, K, dtype=torch.float16, device=device)
        value_new = torch.randn(B, 1, HKV, V, dtype=torch.float16, device=device)
        cos_new = torch.randn(B, 1, K, dtype=torch.float16, device=device)
        sin_new = torch.randn(B, 1, K, dtype=torch.float16, device=device)

        # 拼接到完整的 cos/sin
        cos_full = torch.cat([cos, cos_new], dim=1)
        sin_full = torch.cat([sin, sin_new], dim=1)

        cache_kwargs_new = {
            "cos": cos_full,
            "sin": sin_full,
            "cache_position": torch.arange(T + 1, device=device),
        }

        key_out, value_out = cache.update(
            key_new, value_new, layer_idx=0, cache_kwargs=cache_kwargs_new
        )
        print(f"✓ Decode successful!")
        print(f"  Cache seq length: {cache.get_seq_length()}")
        print(f"  Quantized length: {cache.get_quantized_len()}")
        print(f"  Current length: {cache.get_current_len()}")
    except Exception as e:
        print(f"✗ Decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_fused_integration()
    exit(0 if success else 1)

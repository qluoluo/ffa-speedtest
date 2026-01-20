#!/bin/bash
# Q2FP8 Unified 优化版本 - 快速开始脚本

set -e

echo "=========================================="
echo "Q2FP8 Unified 优化版本 - 快速开始"
echo "=========================================="
echo ""

# 检查当前目录
if [ ! -d "e2e/q2fp8-unified-optimized" ]; then
    echo "错误：请在 ffa-speedtest 根目录下运行此脚本"
    exit 1
fi

echo "步骤 1: 检查依赖..."
python3 -c "import torch; import triton; import transformers; print('✓ 依赖检查通过')" || {
    echo "错误：缺少必要的依赖"
    echo "请安装：pip install torch triton transformers"
    exit 1
}

echo ""
echo "步骤 2: 应用补丁..."
echo "请手动应用以下补丁文件中的修改："
echo "  - PATCH_q2fp8_cache.py"
echo "  - PATCH_attn_kernel.py"
echo "  - PATCH_modeling_llama.py"
echo ""
echo "或者使用以下命令复制原文件并手动修改："
echo ""
echo "  # 复制文件"
echo "  cp e2e/q2fp8-unified/ffa_model/q2fp8_cache.py \\"
echo "     e2e/q2fp8-unified-optimized/ffa_model/q2fp8_cache_optimized.py"
echo ""
echo "  cp e2e/q2fp8-unified/attn_kernel/attn_q2fp8_unified.py \\"
echo "     e2e/q2fp8-unified-optimized/attn_kernel/attn_q2fp8_unified_optimized.py"
echo ""
echo "  cp e2e/q2fp8-unified/ffa_model/modeling_llama.py \\"
echo "     e2e/q2fp8-unified-optimized/ffa_model/modeling_llama_optimized.py"
echo ""
echo "  # 然后根据 PATCH_*.py 文件中的说明应用修改"
echo ""

read -p "是否已完成补丁应用？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请先应用补丁后再继续"
    exit 1
fi

echo ""
echo "步骤 3: 测试导入..."
python3 -c "
import sys
sys.path.insert(0, 'e2e/q2fp8-unified-optimized/ffa_model')
sys.path.insert(0, 'e2e/q2fp8-unified-optimized/attn_kernel')

try:
    from q2fp8_cache_optimized import Q2FP8SymCache
    print('✓ q2fp8_cache_optimized 导入成功')
except Exception as e:
    print(f'✗ q2fp8_cache_optimized 导入失败: {e}')
    sys.exit(1)

try:
    from attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8
    print('✓ attn_q2fp8_unified_optimized 导入成功')
except Exception as e:
    print(f'✗ attn_q2fp8_unified_optimized 导入失败: {e}')
    sys.exit(1)

try:
    from modeling_llama_optimized import LlamaForCausalLM
    print('✓ modeling_llama_optimized 导入成功')
except Exception as e:
    print(f'✗ modeling_llama_optimized 导入失败: {e}')
    sys.exit(1)

print('')
print('所有模块导入成功！')
" || {
    echo ""
    echo "错误：模块导入失败"
    echo "请检查补丁是否正确应用"
    exit 1
}

echo ""
echo "步骤 4: 创建测试脚本..."
cat > e2e/q2fp8-unified-optimized/test_optimized.py << 'EOF'
#!/usr/bin/env python3
"""
Q2FP8 Unified 优化版本 - 简单测试
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "ffa_model"))
sys.path.insert(0, str(Path(__file__).parent / "attn_kernel"))

import torch
from q2fp8_cache_optimized import Q2FP8SymCache

def test_cache_buffer_allocation():
    """测试 cache buffer 预分配"""
    print("测试 1: Cache Buffer 预分配")

    cache = Q2FP8SymCache(
        BS=128,
        use_fp8_residual=True,
        k_bits=2,
        max_decode_tokens=1024,
    )

    # 模拟 prefill
    B, T_prefill, HKV, K = 1, 512, 8, 128
    V = 128
    K_packed = 32  # 2-bit: K / 4

    key_states = torch.randn(B, T_prefill, HKV, K, dtype=torch.float16, device='cuda')
    value_states = torch.randn(B, T_prefill, HKV, V, dtype=torch.float16, device='cuda')

    # Update (prefill)
    cache.update(key_states, value_states, 0, cache_kwargs={})

    # 检查 buffer 是否初始化
    layer = cache.layers[0]
    assert layer.buffer_initialized, "Buffer 应该已初始化"
    assert layer.k_q_buffer is not None, "k_q_buffer 应该已分配"
    assert layer.v_buffer is not None, "v_buffer 应该已分配"

    expected_capacity = T_prefill + 1024
    assert layer.k_q_buffer.shape[1] == expected_capacity, \
        f"Buffer 容量应该是 {expected_capacity}"

    print(f"  ✓ Buffer 已初始化: capacity={expected_capacity}, quantized_len={layer.quantized_len}")

    # 模拟 decode
    for i in range(10):
        key_states = torch.randn(B, 1, HKV, K, dtype=torch.float16, device='cuda')
        value_states = torch.randn(B, 1, HKV, V, dtype=torch.float16, device='cuda')
        cache.update(key_states, value_states, 0, cache_kwargs={})

    print(f"  ✓ Decode 10 steps: quantized_len={layer.quantized_len}, value_len={layer.value_len}")
    print("  ✓ 测试通过！\n")


def test_cudagraph_runner():
    """测试 CUDA Graph runner"""
    print("测试 2: CUDA Graph Runner")

    try:
        from attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        print("  请确保已应用 PATCH_attn_kernel.py 中的修改")
        return

    # 创建测试数据
    B, HQ, HKV, K, V = 1, 32, 8, 128, 128
    T_buffer = 1024
    K_packed = 32

    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device='cuda')
    k_q_buffer = torch.randint(0, 255, (B, T_buffer, HKV, K_packed), dtype=torch.uint8, device='cuda')
    k_scale_buffer = torch.randn(B, T_buffer // 128, HKV, K, dtype=torch.float16, device='cuda')
    v_buffer = torch.randn(B, T_buffer, HKV, V, dtype=torch.float16, device='cuda')
    k_residual_buffer = torch.randn(B, T_buffer, HKV, K, dtype=torch.float8_e4m3fn, device='cuda')
    k_current = torch.randn(B, 128, HKV, K, dtype=torch.float16, device='cuda')
    v_current = torch.randn(B, 128, HKV, V, dtype=torch.float16, device='cuda')

    # 创建 runner
    try:
        runner = CUDAGraphDecodeRunnerQ2FP8(
            q=q,
            k_q=k_q_buffer,
            k_scale=k_scale_buffer,
            v=v_buffer,
            k_current=k_current,
            v_current=v_current,
            current_len=0,
            k_residual=k_residual_buffer,
            quantized_len=512,  # 初始有效长度
            k_bits=2,
            scale=1.0 / (K ** 0.5),
            BS=128,
            SBS=128,
            delta=5.0,
            use_fp8_residual=True,
            max_current=128,
            warmup=2,
        )
        print("  ✓ CUDA Graph runner 创建成功")
    except Exception as e:
        print(f"  ✗ CUDA Graph runner 创建失败: {e}")
        return

    # 测试 replay
    try:
        output = runner.replay(
            q=q,
            k_q=k_q_buffer,
            k_scale=k_scale_buffer,
            v=v_buffer,
            k_current=k_current,
            v_current=v_current,
            current_len=0,
            k_residual=k_residual_buffer,
            quantized_len=512,
        )
        print(f"  ✓ Replay 成功: output shape={output.shape}")

        # 测试动态长度
        output = runner.replay(
            q=q,
            k_q=k_q_buffer,
            k_scale=k_scale_buffer,
            v=v_buffer,
            k_current=k_current,
            v_current=v_current,
            current_len=0,
            k_residual=k_residual_buffer,
            quantized_len=640,  # 新长度
        )
        print(f"  ✓ 动态长度测试成功: quantized_len=640")
        print("  ✓ 测试通过！\n")
    except Exception as e:
        print(f"  ✗ Replay 失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Q2FP8 Unified 优化版本 - 单元测试")
    print("=" * 60)
    print()

    test_cache_buffer_allocation()
    test_cudagraph_runner()

    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
EOF

chmod +x e2e/q2fp8-unified-optimized/test_optimized.py

echo "✓ 测试脚本已创建: e2e/q2fp8-unified-optimized/test_optimized.py"

echo ""
echo "步骤 5: 运行测试..."
python3 e2e/q2fp8-unified-optimized/test_optimized.py || {
    echo ""
    echo "警告：测试失败，请检查实现"
    echo "这可能是因为补丁尚未完全应用"
}

echo ""
echo "=========================================="
echo "快速开始完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果测试失败，请根据 PATCH_*.py 文件完善实现"
echo "2. 运行 E2E benchmark："
echo "   python e2e/benchmark_prefill_decode.py \\"
echo "       --model_path /path/to/model \\"
echo "       --prompt_lengths 16384 \\"
echo "       --decode_lengths 256"
echo ""
echo "3. 查看详细文档："
echo "   - README.md: 概述"
echo "   - IMPLEMENTATION_GUIDE.md: 实现指南"
echo "   - PATCH_*.py: 具体修改说明"
echo ""

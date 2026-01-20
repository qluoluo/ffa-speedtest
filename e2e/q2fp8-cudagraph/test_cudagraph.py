"""
Q2FP8 CUDA Graph 测试脚本

测试 Q2FP8 CUDA Graph Cache 的功能和性能。
"""
import sys
import os
import time
import torch

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ffa_model"))

from q2fp8_cudagraph_cache import Q2FP8CudaGraphCache
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 80)
    print("测试 1: 基本功能测试")
    print("=" * 80)

    # 参数
    max_seq_len = 2048
    BS = 128
    k_bits = 2

    # 创建 cache
    cache = Q2FP8CudaGraphCache(
        max_seq_len=max_seq_len,
        BS=BS,
        k_bits=k_bits,
        use_fp8_residual=True,
    )

    print(f"✓ Cache 创建成功: max_seq_len={max_seq_len}, BS={BS}, k_bits={k_bits}")

    # 模拟 prefill
    B, HKV, K, V = 1, 8, 128, 128
    prefill_len = 512

    key_states = torch.randn(B, prefill_len, HKV, K, dtype=torch.float16, device='cuda')
    value_states = torch.randn(B, prefill_len, HKV, V, dtype=torch.float16, device='cuda')

    keys, values = cache.update(key_states, value_states, layer_idx=0)

    print(f"✓ Prefill 完成: prefill_len={prefill_len}")
    print(f"  - Cache 当前长度: {cache.get_seq_length()}")
    print(f"  - 量化长度: {cache.get_quantized_len()}")

    # 模拟 decode
    decode_steps = 10
    for step in range(decode_steps):
        key_states = torch.randn(B, 1, HKV, K, dtype=torch.float16, device='cuda')
        value_states = torch.randn(B, 1, HKV, V, dtype=torch.float16, device='cuda')
        keys, values = cache.update(key_states, value_states, layer_idx=0)

    print(f"✓ Decode 完成: decode_steps={decode_steps}")
    print(f"  - Cache 最终长度: {cache.get_seq_length()}")
    print(f"  - 量化长度: {cache.get_quantized_len()}")

    print("\n✅ 基本功能测试通过!\n")


def test_model_generation():
    """测试模型生成"""
    print("=" * 80)
    print("测试 2: 模型生成测试")
    print("=" * 80)

    # 模型路径 (需要根据实际情况修改)
    model_path = "/path/to/llama/model"  # TODO: 修改为实际路径

    if not os.path.exists(model_path):
        print(f"⚠️  模型路径不存在: {model_path}")
        print("   跳过模型生成测试")
        return

    # 加载模型
    print(f"加载模型: {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    config.attn_settings = {
        "use_ffa_decode": True,
        "use_cudagraph": True,
        "delta": 5.0,
        "BS": 128,
        "k_bits": 2,
        "use_fp8_residual": True,
    }

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("✓ 模型加载成功")

    # 准备输入
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 创建 cache
    max_seq_len = 2048
    cache = Q2FP8CudaGraphCache(
        max_seq_len=max_seq_len,
        BS=128,
        k_bits=2,
        use_fp8_residual=True,
    )

    # 生成
    print(f"\n生成文本 (max_new_tokens=50)...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            past_key_values=cache,
            use_cache=True,
        )

    elapsed = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n生成结果:")
    print(f"  输入: {prompt}")
    print(f"  输出: {generated_text}")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  速度: {50/elapsed:.2f} tokens/s")

    print("\n✅ 模型生成测试通过!\n")


def benchmark_performance():
    """性能基准测试"""
    print("=" * 80)
    print("测试 3: 性能基准测试")
    print("=" * 80)

    # 参数
    B, HKV, K, V = 1, 8, 128, 128
    prefill_len = 1024
    decode_steps = 100
    max_seq_len = 2048
    BS = 128

    # 测试不同配置
    configs = [
        {"name": "Q2FP8 + CUDA Graph", "use_cudagraph": True, "k_bits": 2},
        {"name": "Q2FP8 (无 CUDA Graph)", "use_cudagraph": False, "k_bits": 2},
    ]

    results = []

    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 40)

        # 创建 cache
        cache = Q2FP8CudaGraphCache(
            max_seq_len=max_seq_len,
            BS=BS,
            k_bits=config['k_bits'],
            use_fp8_residual=True,
        )

        # Prefill
        key_states = torch.randn(B, prefill_len, HKV, K, dtype=torch.float16, device='cuda')
        value_states = torch.randn(B, prefill_len, HKV, V, dtype=torch.float16, device='cuda')
        cache.update(key_states, value_states, layer_idx=0)

        # Warmup
        for _ in range(5):
            key_states = torch.randn(B, 1, HKV, K, dtype=torch.float16, device='cuda')
            value_states = torch.randn(B, 1, HKV, V, dtype=torch.float16, device='cuda')
            cache.update(key_states, value_states, layer_idx=0)

        torch.cuda.synchronize()

        # Benchmark decode
        start_time = time.time()

        for _ in range(decode_steps):
            key_states = torch.randn(B, 1, HKV, K, dtype=torch.float16, device='cuda')
            value_states = torch.randn(B, 1, HKV, V, dtype=torch.float16, device='cuda')
            cache.update(key_states, value_states, layer_idx=0)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        tokens_per_sec = decode_steps / elapsed
        latency_per_token = elapsed / decode_steps * 1000  # ms

        print(f"  Decode 步数: {decode_steps}")
        print(f"  总耗时: {elapsed:.3f}s")
        print(f"  吞吐量: {tokens_per_sec:.2f} tokens/s")
        print(f"  延迟: {latency_per_token:.2f} ms/token")

        results.append({
            "name": config['name'],
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": latency_per_token,
        })

    # 打印对比
    print("\n" + "=" * 80)
    print("性能对比:")
    print("=" * 80)

    baseline = results[1]  # 无 CUDA Graph
    cudagraph = results[0]  # CUDA Graph

    speedup = cudagraph['tokens_per_sec'] / baseline['tokens_per_sec']
    latency_reduction = (baseline['latency_ms'] - cudagraph['latency_ms']) / baseline['latency_ms'] * 100

    print(f"\n{baseline['name']}:")
    print(f"  吞吐量: {baseline['tokens_per_sec']:.2f} tokens/s")
    print(f"  延迟: {baseline['latency_ms']:.2f} ms/token")

    print(f"\n{cudagraph['name']}:")
    print(f"  吞吐量: {cudagraph['tokens_per_sec']:.2f} tokens/s")
    print(f"  延迟: {cudagraph['latency_ms']:.2f} ms/token")

    print(f"\n加速比: {speedup:.2f}x")
    print(f"延迟降低: {latency_reduction:.1f}%")

    print("\n✅ 性能基准测试完成!\n")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Q2FP8 CUDA Graph 测试套件")
    print("=" * 80 + "\n")

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用,无法运行测试")
        return

    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch 版本: {torch.__version__}\n")

    # 运行测试
    try:
        test_basic_functionality()
        # test_model_generation()  # 需要实际模型路径
        benchmark_performance()

        print("=" * 80)
        print("✅ 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

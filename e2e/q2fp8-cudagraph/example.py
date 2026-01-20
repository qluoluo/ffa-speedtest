"""
Q2FP8 CUDA Graph 简单使用示例

展示如何使用 Q2FP8 CUDA Graph Cache 进行文本生成。
"""
import torch
from transformers import AutoTokenizer, AutoConfig

# 导入自定义模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ffa_model"))

from q2fp8_cudagraph_cache import Q2FP8CudaGraphCache
from modeling_llama import LlamaForCausalLM


def main():
    """主函数"""
    print("=" * 80)
    print("Q2FP8 CUDA Graph 使用示例")
    print("=" * 80)

    # ========== 配置参数 ==========
    model_path = "/path/to/your/llama/model"  # TODO: 修改为实际模型路径
    max_seq_len = 4096  # 最大序列长度
    max_new_tokens = 100  # 生成的 token 数量

    # Attention 配置
    attn_config = {
        "use_ffa_decode": True,      # 启用 FFA decode
        "use_cudagraph": True,        # 启用 CUDA Graph
        "delta": 5.0,                 # 阈值偏移
        "BS": 128,                    # Block size
        "k_bits": 2,                  # 量化位数 (2 或 4)
        "use_fp8_residual": True,     # 使用 FP8 残差
    }

    # ========== 检查环境 ==========
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return

    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch: {torch.__version__}")

    if not os.path.exists(model_path):
        print(f"\n⚠️  模型路径不存在: {model_path}")
        print("   请修改 model_path 为实际的模型路径")
        print("\n示例:")
        print("   model_path = '/data/models/Llama-3.1-8B'")
        return

    # ========== 加载模型 ==========
    print(f"\n加载模型: {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    config.attn_settings = attn_config

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ 模型加载成功")

    # ========== 创建 Cache ==========
    print(f"\n创建 Cache (max_seq_len={max_seq_len})")

    cache = Q2FP8CudaGraphCache(
        max_seq_len=max_seq_len,
        BS=attn_config["BS"],
        k_bits=attn_config["k_bits"],
        use_fp8_residual=attn_config["use_fp8_residual"],
    )

    print("✓ Cache 创建成功")

    # ========== 生成文本 ==========
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence is",
        "In a world where technology",
    ]

    print("\n" + "=" * 80)
    print("开始生成")
    print("=" * 80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("-" * 80)

        # 重置 cache (每个 prompt 独立)
        cache.reset()

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 生成
        import time
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,  # 贪婪解码
            )

        elapsed = time.time() - start_time

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 打印结果
        print(f"生成结果:\n{generated_text}\n")
        print(f"统计:")
        print(f"  - 生成 tokens: {max_new_tokens}")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 速度: {max_new_tokens/elapsed:.2f} tokens/s")
        print(f"  - Cache 长度: {cache.get_seq_length()}")

    print("\n" + "=" * 80)
    print("✅ 生成完成!")
    print("=" * 80)

    # ========== 性能提示 ==========
    print("\n性能优化提示:")
    print("1. 调整 delta 参数 (3.0-10.0) 平衡速度和精度")
    print("2. 选择合适的 max_seq_len 避免显存浪费")
    print("3. 使用 k_bits=2 获得更高压缩比")
    print("4. CUDA Graph 在首次调用时录制,后续调用会更快")


if __name__ == "__main__":
    main()

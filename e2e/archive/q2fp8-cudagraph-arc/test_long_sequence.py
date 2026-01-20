"""
长序列测试脚本: 16K 和 32K

测试全局共享 CUDA Graph 在长序列场景下的性能。

对比三种配置:
1. Baseline (Flash Attention 2)
2. Q2FP8 + 每层独立 CUDA Graph
3. Q2FP8 + 全局共享 CUDA Graph
"""
import sys
import os
import time
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ffa_model"))

from modeling_llama import LlamaForCausalLM
from q2fp8_cudagraph_cache import Q2FP8CudaGraphCache
from global_cudagraph_manager import GlobalCudaGraphManager


def generate_long_prompt(tokenizer, target_length):
    """生成指定长度的 prompt"""
    # 使用重复的文本生成长 prompt
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base_text, add_special_tokens=False)

    # 重复直到达到目标长度
    while len(tokens) < target_length:
        tokens.extend(tokens[:min(len(tokens), target_length - len(tokens))])

    return tokens[:target_length]


def benchmark_config(
    model_path,
    prompt_length,
    decode_length,
    config_name,
    use_ffa=False,
    use_cudagraph=False,
    use_global_cudagraph=False,
    num_runs=3,
):
    """
    测试单个配置的性能。

    Args:
        model_path: 模型路径
        prompt_length: prompt 长度
        decode_length: decode token 数量
        config_name: 配置名称
        use_ffa: 是否使用 FFA
        use_cudagraph: 是否使用 CUDA Graph
        use_global_cudagraph: 是否使用全局共享 CUDA Graph
        num_runs: 运行次数
    """
    print(f"\n{'='*80}")
    print(f"测试配置: {config_name}")
    print(f"  Prompt长度: {prompt_length}, Decode长度: {decode_length}")
    print(f"  FFA: {use_ffa}, CUDA Graph: {use_cudagraph}, 全局共享: {use_global_cudagraph}")
    print(f"{'='*80}\n")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for run_idx in range(num_runs):
        print(f"运行 {run_idx + 1}/{num_runs}...")

        # 重置全局 CUDA Graph Manager
        if use_global_cudagraph:
            GlobalCudaGraphManager.reset_instance()

        # 加载模型
        config = AutoConfig.from_pretrained(model_path)

        if use_ffa:
            config.attn_settings = {
                "use_ffa_decode": True,
                "use_cudagraph": use_cudagraph,
                "use_global_cudagraph": use_global_cudagraph,
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

        # 生成 prompt
        prompt_tokens = generate_long_prompt(tokenizer, prompt_length)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device='cuda')

        # 创建 cache
        if use_ffa:
            cache = Q2FP8CudaGraphCache(
                max_seq_len=prompt_length + decode_length + 1024,  # 留一些余量
                BS=128,
                k_bits=2,
                use_fp8_residual=True,
            )
        else:
            cache = None

        # Warmup
        print("  Warmup...")
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=5,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
            )

        # 重新创建 cache 用于实际测试
        if use_ffa:
            cache = Q2FP8CudaGraphCache(
                max_seq_len=prompt_length + decode_length + 1024,
                BS=128,
                k_bits=2,
                use_fp8_residual=True,
            )

        # 测试 prefill
        print("  测试 Prefill...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=cache,
                use_cache=True,
            )

        torch.cuda.synchronize()
        prefill_time = (time.time() - prefill_start) * 1000  # ms

        # 测试 decode
        print("  测试 Decode...")
        decode_times = []

        with torch.no_grad():
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for step in range(decode_length):
                torch.cuda.synchronize()
                step_start = time.time()

                outputs = model(
                    next_token_id,
                    past_key_values=cache,
                    use_cache=True,
                )

                torch.cuda.synchronize()
                step_time = (time.time() - step_start) * 1000  # ms
                decode_times.append(step_time)

                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 统计
        total_decode_time = sum(decode_times)
        avg_decode_time = total_decode_time / len(decode_times)
        first_token_time = decode_times[0]
        subsequent_avg = sum(decode_times[1:]) / len(decode_times[1:]) if len(decode_times) > 1 else 0

        result = {
            "run": run_idx + 1,
            "prefill_time_ms": prefill_time,
            "total_decode_time_ms": total_decode_time,
            "avg_decode_time_ms": avg_decode_time,
            "first_token_time_ms": first_token_time,
            "subsequent_avg_ms": subsequent_avg,
            "throughput_tokens_per_sec": 1000.0 / avg_decode_time if avg_decode_time > 0 else 0,
        }
        results.append(result)

        print(f"    Prefill: {prefill_time:.2f} ms")
        print(f"    首个token: {first_token_time:.2f} ms")
        print(f"    后续平均: {subsequent_avg:.2f} ms/token")
        print(f"    总体平均: {avg_decode_time:.2f} ms/token")
        print(f"    吞吐量: {result['throughput_tokens_per_sec']:.2f} tokens/s")

        # 清理
        del model
        del cache
        torch.cuda.empty_cache()

    # 计算平均值
    avg_result = {
        "config_name": config_name,
        "prompt_length": prompt_length,
        "decode_length": decode_length,
        "num_runs": num_runs,
        "avg_prefill_ms": sum(r["prefill_time_ms"] for r in results) / num_runs,
        "avg_decode_ms_per_token": sum(r["avg_decode_time_ms"] for r in results) / num_runs,
        "avg_first_token_ms": sum(r["first_token_time_ms"] for r in results) / num_runs,
        "avg_subsequent_ms": sum(r["subsequent_avg_ms"] for r in results) / num_runs,
        "avg_throughput": sum(r["throughput_tokens_per_sec"] for r in results) / num_runs,
        "runs": results,
    }

    print(f"\n平均结果:")
    print(f"  Prefill: {avg_result['avg_prefill_ms']:.2f} ms")
    print(f"  首个token: {avg_result['avg_first_token_ms']:.2f} ms")
    print(f"  后续平均: {avg_result['avg_subsequent_ms']:.2f} ms/token")
    print(f"  总体平均: {avg_result['avg_decode_ms_per_token']:.2f} ms/token")
    print(f"  吞吐量: {avg_result['avg_throughput']:.2f} tokens/s")

    return avg_result


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="长序列性能测试")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt_lengths", type=str, default="16384,32768", help="Prompt长度列表,逗号分隔")
    parser.add_argument("--decode_length", type=int, default=128, help="Decode token数量")
    parser.add_argument("--num_runs", type=int, default=3, help="每个配置运行次数")
    parser.add_argument("--output_dir", type=str, default="outputs_long_seq", help="输出目录")
    args = parser.parse_args()

    # 解析 prompt 长度
    prompt_lengths = [int(x.strip()) for x in args.prompt_lengths.split(",")]

    print(f"\n{'='*80}")
    print(f"长序列性能测试")
    print(f"{'='*80}")
    print(f"模型: {args.model_path}")
    print(f"Prompt长度: {prompt_lengths}")
    print(f"Decode长度: {args.decode_length}")
    print(f"运行次数: {args.num_runs}")
    print(f"{'='*80}\n")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for prompt_length in prompt_lengths:
        print(f"\n\n{'#'*80}")
        print(f"# 测试 Prompt 长度: {prompt_length}")
        print(f"{'#'*80}\n")

        # 配置1: Baseline (Flash Attention 2)
        result_baseline = benchmark_config(
            model_path=args.model_path,
            prompt_length=prompt_length,
            decode_length=args.decode_length,
            config_name="Baseline (Flash Attention 2)",
            use_ffa=False,
            use_cudagraph=False,
            use_global_cudagraph=False,
            num_runs=args.num_runs,
        )
        all_results.append(result_baseline)

        # 配置2: Q2FP8 + 每层独立 CUDA Graph
        result_per_layer = benchmark_config(
            model_path=args.model_path,
            prompt_length=prompt_length,
            decode_length=args.decode_length,
            config_name="Q2FP8 + Per-Layer CUDA Graph",
            use_ffa=True,
            use_cudagraph=True,
            use_global_cudagraph=False,
            num_runs=args.num_runs,
        )
        all_results.append(result_per_layer)

        # 配置3: Q2FP8 + 全局共享 CUDA Graph
        result_global = benchmark_config(
            model_path=args.model_path,
            prompt_length=prompt_length,
            decode_length=args.decode_length,
            config_name="Q2FP8 + Global Shared CUDA Graph",
            use_ffa=True,
            use_cudagraph=True,
            use_global_cudagraph=True,
            num_runs=args.num_runs,
        )
        all_results.append(result_global)

        # 打印对比
        print(f"\n{'='*80}")
        print(f"Prompt {prompt_length} 性能对比")
        print(f"{'='*80}")
        print(f"{'配置':<40} {'Decode (ms/token)':<20} {'吞吐量 (tokens/s)':<20}")
        print(f"{'-'*80}")
        print(f"{result_baseline['config_name']:<40} {result_baseline['avg_decode_ms_per_token']:<20.2f} {result_baseline['avg_throughput']:<20.2f}")
        print(f"{result_per_layer['config_name']:<40} {result_per_layer['avg_decode_ms_per_token']:<20.2f} {result_per_layer['avg_throughput']:<20.2f}")
        print(f"{result_global['config_name']:<40} {result_global['avg_decode_ms_per_token']:<20.2f} {result_global['avg_throughput']:<20.2f}")
        print(f"{'-'*80}")

        speedup_per_layer = result_baseline['avg_decode_ms_per_token'] / result_per_layer['avg_decode_ms_per_token']
        speedup_global = result_baseline['avg_decode_ms_per_token'] / result_global['avg_decode_ms_per_token']

        print(f"Per-Layer vs Baseline: {speedup_per_layer:.2f}x")
        print(f"Global vs Baseline: {speedup_global:.2f}x")
        print(f"{'='*80}\n")

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"long_sequence_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 生成总结报告
    summary_file = output_dir / f"SUMMARY_{timestamp}.md"
    with open(summary_file, 'w') as f:
        f.write("# 长序列性能测试总结\n\n")
        f.write(f"**测试日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**模型**: {args.model_path}\n\n")
        f.write(f"**Decode长度**: {args.decode_length} tokens\n\n")
        f.write(f"**运行次数**: {args.num_runs}\n\n")

        f.write("## 测试结果\n\n")

        for prompt_length in prompt_lengths:
            f.write(f"### Prompt 长度: {prompt_length}\n\n")
            f.write("| 配置 | Decode (ms/token) | 吞吐量 (tokens/s) | vs Baseline |\n")
            f.write("|------|-------------------|-------------------|-------------|\n")

            results_for_length = [r for r in all_results if r['prompt_length'] == prompt_length]
            baseline = results_for_length[0]

            for result in results_for_length:
                speedup = baseline['avg_decode_ms_per_token'] / result['avg_decode_ms_per_token']
                f.write(f"| {result['config_name']} | {result['avg_decode_ms_per_token']:.2f} | {result['avg_throughput']:.2f} | {speedup:.2f}x |\n")

            f.write("\n")

    print(f"总结报告已保存到: {summary_file}")


if __name__ == "__main__":
    main()

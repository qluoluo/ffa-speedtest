#!/usr/bin/env python3
"""
端到端速度测试对比脚本
比较 Quest 和 FFA-Q2FP8-Sym 的性能

用法:
    python run_all.py --prompt_type medium --max_new_tokens 128
    python run_all.py --prompt_tokens 2000 --max_new_tokens 256 --detailed
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "shared"))
# Use local model directories
sys.path.insert(0, str(Path(__file__).parent / "quest" / "quest_model"))
sys.path.insert(0, str(Path(__file__).parent / "q2fp8" / "ffa_model"))

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Import benchmark utilities
from benchmark_utils import (
    BenchmarkResult,
    reset_memory_stats,
    measure_peak_memory,
    print_result,
    save_results,
    compare_results,
    Timer,
)
from test_prompts import get_test_prompts, generate_repeated_prompt

# Quest imports
try:
    from quest import LlamaForCausalLM as QuestLlama
    QUEST_AVAILABLE = True
except ImportError:
    print("Warning: Quest not available")
    QUEST_AVAILABLE = False

# FFA imports
try:
    from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
    from q2fp8_cache import Q2FP8SymCache
    FFA_AVAILABLE = True
except ImportError:
    print("Warning: FFA-Q2FP8-Sym not available")
    FFA_AVAILABLE = False


DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"


def benchmark_baseline(
    model_path: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Benchmark vanilla transformers as baseline."""
    print("\n[Baseline] Loading vanilla HuggingFace model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="flash_attention_2",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Warmup
    warmup_inputs = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    total_times = []
    generated_tokens_list = []

    for _ in range(num_runs):
        reset_memory_stats()

        total_timer = Timer()
        total_timer.start()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        total_timer.stop()
        total_times.append(total_timer.elapsed_ms)
        generated_tokens_list.append(outputs.shape[1] - prompt_length)

    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = int(sum(generated_tokens_list) / len(generated_tokens_list))

    # Estimate prefill/decode
    estimated_prefill_ratio = min(0.25, prompt_length / (prompt_length + generated_tokens * 8))
    prefill_time = avg_total_time * estimated_prefill_ratio
    decode_time = avg_total_time - prefill_time

    prefill_throughput = prompt_length / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_throughput = generated_tokens / (decode_time / 1000) if decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        method_name="HuggingFace (baseline)",
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_time_ms=prefill_time,
        decode_time_ms=decode_time,
        total_time_ms=avg_total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        total_throughput=total_throughput,
        memory_peak_mb=memory_peak,
        config={"attn_implementation": "flash_attention_2"},
    )


def benchmark_quest(
    model_path: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    page_size: int = 16,
    token_budget: int = 1024,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Benchmark Quest."""
    print("\n[Quest] Loading Quest model...")

    model = QuestLlama.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Initialize generation_config if it's None
    if model.generation_config is None:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)

    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    cfg_max_len = getattr(model.config, "max_position_embeddings", None)
    max_seq_len = min(cfg_max_len, prompt_length + max_new_tokens) if cfg_max_len else (prompt_length + max_new_tokens)

    # Warmup
    warmup_inputs = tokenizer("Hello", return_tensors="pt").to(device)
    model.quest_init(page_size=16, max_seq_len=100, token_budget=64, dtype=dtype, device=device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    model.quest_clear()
    # Clear iController to allow re-initialization
    model.model.iController = None

    total_times = []
    generated_tokens_list = []

    config = {
        "page_size": page_size,
        "token_budget": token_budget,
        "max_seq_len": max_seq_len,
    }

    for _ in range(num_runs):
        reset_memory_stats()

        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
        )

        total_timer = Timer()
        total_timer.start()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        total_timer.stop()
        model.quest_clear()
        # Clear iController to allow re-initialization
        model.model.iController = None

        total_times.append(total_timer.elapsed_ms)
        generated_tokens_list.append(outputs.shape[1] - prompt_length)

    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = int(sum(generated_tokens_list) / len(generated_tokens_list))

    # Estimate prefill/decode
    estimated_prefill_ratio = min(0.25, prompt_length / (prompt_length + generated_tokens * 8))
    prefill_time = avg_total_time * estimated_prefill_ratio
    decode_time = avg_total_time - prefill_time

    prefill_throughput = prompt_length / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_throughput = generated_tokens / (decode_time / 1000) if decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        method_name=f"Quest (budget={token_budget})",
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_time_ms=prefill_time,
        decode_time_ms=decode_time,
        total_time_ms=avg_total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        total_throughput=total_throughput,
        memory_peak_mb=memory_peak,
        config=config,
    )


def benchmark_ffa(
    model_path: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int = 128,
    k_bits: int = 2,
    use_fp8_residual: bool = True,
    delta: float = 5.0,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Benchmark FFA-Q2FP8-Sym."""
    print("\n[FFA-Q2FP8] Loading FFA-Q2FP8-Sym model...")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": True,
        "delta": delta,
        "BS": block_size,
        "SBS": block_size,
        "use_fp8_residual": use_fp8_residual,
        "k_bits": k_bits,
    }

    model = FFALlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="flash_attention_2",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Warmup
    warmup_inputs = tokenizer("Hello", return_tensors="pt").to(device)
    warmup_cache = Q2FP8SymCache(BS=block_size, use_fp8_residual=use_fp8_residual, k_bits=k_bits)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=3, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id, past_key_values=warmup_cache)

    total_times = []
    generated_tokens_list = []

    benchmark_config = {
        "block_size": block_size,
        "k_bits": k_bits,
        "use_fp8_residual": use_fp8_residual,
        "delta": delta,
    }

    for _ in range(num_runs):
        reset_memory_stats()

        cache = Q2FP8SymCache(BS=block_size, use_fp8_residual=use_fp8_residual, k_bits=k_bits)

        total_timer = Timer()
        total_timer.start()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache,
            )

        total_timer.stop()

        total_times.append(total_timer.elapsed_ms)
        generated_tokens_list.append(outputs.shape[1] - prompt_length)

    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = int(sum(generated_tokens_list) / len(generated_tokens_list))

    # Estimate prefill/decode
    estimated_prefill_ratio = min(0.25, prompt_length / (prompt_length + generated_tokens * 8))
    prefill_time = avg_total_time * estimated_prefill_ratio
    decode_time = avg_total_time - prefill_time

    prefill_throughput = prompt_length / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_throughput = generated_tokens / (decode_time / 1000) if decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        method_name=f"FFA-Q2FP8 (k_bits={k_bits})",
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_time_ms=prefill_time,
        decode_time_ms=decode_time,
        total_time_ms=avg_total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        total_throughput=total_throughput,
        memory_peak_mb=memory_peak,
        config=benchmark_config,
    )


def main():
    parser = argparse.ArgumentParser(description="End-to-end speed comparison: Quest vs FFA-Q2FP8-Sym")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to the model")
    parser.add_argument("--prompt_type", choices=["short", "medium", "long", "custom"], default="medium")
    parser.add_argument("--prompt_tokens", type=int, default=None, help="Target prompt length in tokens (for custom)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")

    # Quest params
    parser.add_argument("--quest_page_size", type=int, default=16)
    parser.add_argument("--quest_token_budget", type=int, default=1024)

    # FFA params
    parser.add_argument("--ffa_block_size", type=int, default=128)
    parser.add_argument("--ffa_k_bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--ffa_delta", type=float, default=5.0)
    parser.add_argument("--ffa_use_fp8_residual", action="store_true", default=True)

    # Run params
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--include_baseline", action="store_true", help="Include HuggingFace baseline")
    parser.add_argument("--skip_quest", action="store_true", help="Skip Quest benchmark")
    parser.add_argument("--skip_ffa", action="store_true", help="Skip FFA benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 80)
    print("End-to-End Speed Comparison: Quest vs FFA-Q2FP8-Sym")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Number of runs: {args.num_runs}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get prompt
    prompts = get_test_prompts()
    if args.prompt_type == "custom" and args.prompt_tokens:
        prompt = generate_repeated_prompt(prompts["medium"], args.prompt_tokens, tokenizer)
    else:
        prompt = prompts[args.prompt_type]

    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\nActual prompt tokens: {prompt_tokens}")

    results = []

    # Baseline
    if args.include_baseline:
        try:
            result = benchmark_baseline(
                args.model_path, tokenizer, prompt, args.max_new_tokens,
                device, dtype, args.num_runs
            )
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"Baseline benchmark failed: {e}")

    # Quest
    if not args.skip_quest and QUEST_AVAILABLE:
        try:
            result = benchmark_quest(
                args.model_path, tokenizer, prompt, args.max_new_tokens,
                device, dtype,
                page_size=args.quest_page_size,
                token_budget=args.quest_token_budget,
                num_runs=args.num_runs
            )
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"Quest benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # FFA-Q2FP8
    if not args.skip_ffa and FFA_AVAILABLE:
        try:
            result = benchmark_ffa(
                args.model_path, tokenizer, prompt, args.max_new_tokens,
                device, dtype,
                block_size=args.ffa_block_size,
                k_bits=args.ffa_k_bits,
                use_fp8_residual=args.ffa_use_fp8_residual,
                delta=args.ffa_delta,
                num_runs=args.num_runs
            )
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FFA-Q2FP8 benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(results) >= 2:
        compare_results(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_file = args.output
    else:
        output_file = str(Path(__file__).parent / f"comparison_results_{timestamp}.json")

    save_results(results, output_file)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"{result.method_name}: {result.total_time_ms:.2f}ms total, "
              f"{result.decode_throughput:.2f} decode tok/s, "
              f"{result.memory_peak_mb:.1f}MB peak memory")
    print("=" * 80)


if __name__ == "__main__":
    main()

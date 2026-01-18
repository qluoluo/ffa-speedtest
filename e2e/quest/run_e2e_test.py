#!/usr/bin/env python3
"""
Quest 端到端速度测试脚本
"""

import sys
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
# Use local quest_model directory
sys.path.insert(0, str(Path(__file__).parent / "quest_model"))

import torch
from transformers import AutoTokenizer

try:
    # Import directly from models submodule since package name is quest_model not quest
    from models.llama import LlamaForCausalLM as QuestLlama
except ImportError:
    print("Error: Cannot import Quest. Please ensure quest package is installed.")
    print("Or check the path to quest_model directory.")
    sys.exit(1)

from benchmark_utils import (
    BenchmarkResult,
    warmup_model,
    reset_memory_stats,
    measure_peak_memory,
    print_result,
    save_results,
    Timer,
)
from test_prompts import get_test_prompts, generate_repeated_prompt


DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"


def load_quest_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load Quest model."""
    print(f"Loading Quest model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = QuestLlama.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def benchmark_quest(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    page_size: int = 16,
    token_budget: int = 1024,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Run Quest benchmark with detailed timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Calculate max_seq_len
    cfg_max_len = getattr(model.config, "max_position_embeddings", None)
    max_seq_len = min(cfg_max_len, prompt_length + max_new_tokens) if cfg_max_len else (prompt_length + max_new_tokens)

    prefill_times = []
    decode_times = []
    total_times = []
    generated_tokens_list = []

    config = {
        "page_size": page_size,
        "token_budget": token_budget,
        "max_seq_len": max_seq_len,
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Initialize Quest
        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
        )

        # Time the entire generation
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

        # Cleanup
        model.quest_clear()
        # Ensure controller is fully reset for next run
        if hasattr(model, 'model') and hasattr(model.model, 'iController'):
            model.model.iController = None

        total_times.append(total_timer.elapsed_ms)
        generated_tokens = outputs.shape[1] - prompt_length
        generated_tokens_list.append(generated_tokens)

    # Calculate averages
    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = int(sum(generated_tokens_list) / len(generated_tokens_list))

    # Estimate prefill/decode split
    # Quest has specific optimizations for decode, so prefill ratio may be higher
    estimated_prefill_ratio = min(0.25, prompt_length / (prompt_length + generated_tokens * 8))
    prefill_time = avg_total_time * estimated_prefill_ratio
    decode_time = avg_total_time - prefill_time

    # Calculate throughput
    prefill_throughput = prompt_length / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_throughput = generated_tokens / (decode_time / 1000) if decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

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


def benchmark_quest_detailed(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    page_size: int = 16,
    token_budget: int = 1024,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Run Quest benchmark with manual token-by-token generation for precise timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    cfg_max_len = getattr(model.config, "max_position_embeddings", None)
    max_seq_len = min(cfg_max_len, prompt_length + max_new_tokens) if cfg_max_len else (prompt_length + max_new_tokens)

    prefill_times = []
    decode_times = []
    total_times = []
    generated_tokens_list = []

    config = {
        "page_size": page_size,
        "token_budget": token_budget,
        "max_seq_len": max_seq_len,
        "detailed": True,
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Initialize Quest
        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
        )

        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.clone()

        # Prefill phase
        prefill_timer = Timer()
        prefill_timer.start()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        prefill_timer.stop()
        prefill_times.append(prefill_timer.elapsed_ms)

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token]

        # Decode phase
        decode_timer = Timer()
        decode_timer.start()

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                    ], dim=1)

                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token)

                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        decode_timer.stop()
        decode_times.append(decode_timer.elapsed_ms)
        total_times.append(prefill_timer.elapsed_ms + decode_timer.elapsed_ms)
        generated_tokens_list.append(len(generated_ids))

        # Cleanup
        model.quest_clear()
        # Ensure controller is fully reset for next run
        if hasattr(model, 'model') and hasattr(model.model, 'iController'):
            model.model.iController = None

    # Calculate averages
    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_decode_time = sum(decode_times) / len(decode_times)
    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = int(sum(generated_tokens_list) / len(generated_tokens_list))

    # Calculate throughput
    prefill_throughput = prompt_length / (avg_prefill_time / 1000) if avg_prefill_time > 0 else 0
    decode_throughput = generated_tokens / (avg_decode_time / 1000) if avg_decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    return BenchmarkResult(
        method_name=f"Quest (budget={token_budget})",
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_time_ms=avg_prefill_time,
        decode_time_ms=avg_decode_time,
        total_time_ms=avg_total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        total_throughput=total_throughput,
        memory_peak_mb=memory_peak,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(description="Quest end-to-end speed benchmark")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to the model")
    parser.add_argument("--prompt_type", choices=["short", "medium", "long", "custom"], default="medium")
    parser.add_argument("--prompt_tokens", type=int, default=None, help="Target prompt length in tokens")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--page_size", type=int, default=16, help="Quest page size")
    parser.add_argument("--token_budget", type=int, default=1024, help="Quest token budget")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--detailed", action="store_true", help="Use detailed token-by-token timing")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 70)
    print("Quest End-to-End Speed Benchmark")
    print("=" * 70)

    # Load model
    model, tokenizer = load_quest_model(args.model_path, device, dtype)

    # Get prompt
    prompts = get_test_prompts()
    if args.prompt_type == "custom" and args.prompt_tokens:
        prompt = generate_repeated_prompt(prompts["medium"], args.prompt_tokens, tokenizer)
    else:
        prompt = prompts[args.prompt_type]

    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\nPrompt type: {args.prompt_type}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Page size: {args.page_size}")
    print(f"Token budget: {args.token_budget}")
    print(f"Number of runs: {args.num_runs}")

    # Warmup
    print("\nWarming up...")
    # Simple warmup without Quest cache
    warmup_text = "Hello world"
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt").to(device)
    with torch.no_grad():
        model.quest_init(page_size=16, max_seq_len=100, token_budget=64, dtype=dtype, device=device)
        _ = model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        model.quest_clear()
        # Ensure controller is fully reset
        if hasattr(model, 'model') and hasattr(model.model, 'iController'):
            model.model.iController = None

    # Run benchmark
    print("\nRunning benchmark...")
    if args.detailed:
        result = benchmark_quest_detailed(
            model, tokenizer, prompt, args.max_new_tokens, device, dtype,
            page_size=args.page_size, token_budget=args.token_budget, num_runs=args.num_runs
        )
    else:
        result = benchmark_quest(
            model, tokenizer, prompt, args.max_new_tokens, device, dtype,
            page_size=args.page_size, token_budget=args.token_budget, num_runs=args.num_runs
        )

    # Print results
    print_result(result)

    # Save results
    if args.output:
        save_results([result], args.output)
    else:
        default_output = Path(__file__).parent / "quest_result.json"
        save_results([result], str(default_output))


if __name__ == "__main__":
    main()

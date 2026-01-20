#!/usr/bin/env python3
"""
详细对比 Q2FP8 和 Baseline 在不同输入/输出长度下的 prefill 和 decode 性能
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "shared"))

from benchmark_utils import (
    reset_memory_stats,
    measure_peak_memory,
    Timer,
)
from test_prompts import generate_repeated_prompt


DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"


def load_baseline_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load vanilla HuggingFace model with Flash Attention 2."""
    print(f"Loading baseline model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_q2fp8_model(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    delta: float = 5.0,
    block_size: int = 128,
    k_bits: int = 2,
    use_fp8_residual: bool = True,
    max_decode_tokens: int = 1024,
):
    """Load Q2FP8-Page model."""
    model_dir = "q2fp8-page"
    print(f"Loading Q2FP8-Page (Block-wise JIT CUDA Graph) model from {model_path}...")

    sys.path.insert(0, str(Path(__file__).parent / model_dir / "ffa_model"))

    # Compatibility patch
    import transformers.integrations
    if not hasattr(transformers.integrations, 'use_kernel_forward_from_hub'):
        sys.path.insert(0, str(Path(__file__).parent / model_dir))
        from compat_patch import use_kernel_forward_from_hub
        transformers.integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub

    from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
    from q2fp8_cache import Q2FP8SymCache as CacheClass

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": True,
        "delta": delta,
        "BS": block_size,
        "SBS": block_size,
        "use_fp8_residual": use_fp8_residual,
        "k_bits": k_bits,
    }

    print("DEBUG: About to call from_pretrained...")
    model = FFALlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    print("DEBUG: from_pretrained completed, calling model.eval()...")
    model.eval()
    print("DEBUG: model.eval() completed!")

    return model, tokenizer, CacheClass, max_decode_tokens


def benchmark_baseline(
    model,
    tokenizer,
    prompt: str,
    num_decode_tokens: int,
    device: torch.device,
    num_runs: int = 3,
) -> Dict:
    """Benchmark baseline model with separate prefill and decode timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    results = {
        "prompt_length": prompt_length,
        "num_decode_tokens": num_decode_tokens,
        "prefill_times": [],
        "decode_times": [],
        "per_token_times": [],
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.clone()

        # === PREFILL PHASE ===
        prefill_timer = Timer()
        prefill_timer.start()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        torch.cuda.synchronize()
        prefill_timer.stop()
        results["prefill_times"].append(prefill_timer.elapsed_ms)

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # === DECODE PHASE ===
        decode_timer = Timer()
        per_token_times = []

        for step in range(num_decode_tokens):
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)

            token_timer = Timer()
            token_timer.start()

            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            token_timer.stop()
            per_token_times.append(token_timer.elapsed_ms)

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decode_total = sum(per_token_times)
        results["decode_times"].append(decode_total)
        results["per_token_times"].append(per_token_times)

    # Calculate averages
    results["avg_prefill_ms"] = sum(results["prefill_times"]) / len(results["prefill_times"])
    results["avg_decode_ms"] = sum(results["decode_times"]) / len(results["decode_times"])
    results["avg_per_token_ms"] = results["avg_decode_ms"] / num_decode_tokens
    results["memory_mb"] = measure_peak_memory()

    return results


def benchmark_q2fp8(
    model,
    tokenizer,
    Q2FP8SymCache,
    prompt: str,
    num_decode_tokens: int,
    device: torch.device,
    block_size: int = 128,
    k_bits: int = 2,
    use_fp8_residual: bool = True,
    max_decode_tokens: int = 4096,
    num_runs: int = 3,
) -> Dict:
    """Benchmark Q2FP8 model with separate prefill and decode timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    results = {
        "prompt_length": prompt_length,
        "num_decode_tokens": num_decode_tokens,
        "prefill_times": [],
        "decode_times": [],
        "per_token_times": [],
        "quantization_times": [],  # Time spent in quantization during prefill
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Create fresh cache
        cache = Q2FP8SymCache(BS=block_size, use_fp8_residual=use_fp8_residual, k_bits=k_bits)

        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.clone()

        # === PREFILL PHASE ===
        prefill_timer = Timer()
        prefill_timer.start()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

        torch.cuda.synchronize()
        prefill_timer.stop()
        results["prefill_times"].append(prefill_timer.elapsed_ms)

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # === DECODE PHASE ===
        decode_timer = Timer()
        per_token_times = []

        for step in range(num_decode_tokens):
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)

            token_timer = Timer()
            token_timer.start()

            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            token_timer.stop()
            per_token_times.append(token_timer.elapsed_ms)

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decode_total = sum(per_token_times)
        results["decode_times"].append(decode_total)
        results["per_token_times"].append(per_token_times)

    # Calculate averages
    results["avg_prefill_ms"] = sum(results["prefill_times"]) / len(results["prefill_times"])
    results["avg_decode_ms"] = sum(results["decode_times"]) / len(results["decode_times"])
    results["avg_per_token_ms"] = results["avg_decode_ms"] / num_decode_tokens
    results["memory_mb"] = measure_peak_memory()

    return results


def print_comparison(baseline_results: Dict, q2fp8_results: Dict, config: Dict):
    """Print detailed comparison."""
    prompt_len = baseline_results["prompt_length"]
    decode_len = baseline_results["num_decode_tokens"]

    print("\n" + "=" * 100)
    print(f"RESULTS: Prompt Length = {prompt_len}, Decode Tokens = {decode_len}")
    print("=" * 100)

    # Prefill comparison
    print("\n--- PREFILL PHASE ---")
    print(f"{'Metric':<30} {'Baseline':<20} {'Q2FP8':<20} {'Ratio':<15}")
    print("-" * 85)

    baseline_prefill = baseline_results["avg_prefill_ms"]
    q2fp8_prefill = q2fp8_results["avg_prefill_ms"]
    prefill_ratio = baseline_prefill / q2fp8_prefill

    print(f"{'Prefill Time (ms)':<30} {baseline_prefill:<20.2f} {q2fp8_prefill:<20.2f} {prefill_ratio:<15.3f}x")
    print(f"{'Prefill Throughput (tok/s)':<30} {prompt_len/(baseline_prefill/1000):<20.2f} {prompt_len/(q2fp8_prefill/1000):<20.2f} {prefill_ratio:<15.3f}x")

    # Decode comparison
    print("\n--- DECODE PHASE ---")
    print(f"{'Metric':<30} {'Baseline':<20} {'Q2FP8':<20} {'Ratio':<15}")
    print("-" * 85)

    baseline_decode = baseline_results["avg_decode_ms"]
    q2fp8_decode = q2fp8_results["avg_decode_ms"]
    decode_ratio = baseline_decode / q2fp8_decode

    baseline_per_token = baseline_results["avg_per_token_ms"]
    q2fp8_per_token = q2fp8_results["avg_per_token_ms"]

    print(f"{'Total Decode Time (ms)':<30} {baseline_decode:<20.2f} {q2fp8_decode:<20.2f} {decode_ratio:<15.3f}x")
    print(f"{'Per-Token Time (ms)':<30} {baseline_per_token:<20.2f} {q2fp8_per_token:<20.2f} {decode_ratio:<15.3f}x")
    print(f"{'Decode Throughput (tok/s)':<30} {1000/baseline_per_token:<20.2f} {1000/q2fp8_per_token:<20.2f} {decode_ratio:<15.3f}x")

    # Total comparison
    print("\n--- TOTAL (PREFILL + DECODE) ---")
    print(f"{'Metric':<30} {'Baseline':<20} {'Q2FP8':<20} {'Ratio':<15}")
    print("-" * 85)

    baseline_total = baseline_prefill + baseline_decode
    q2fp8_total = q2fp8_prefill + q2fp8_decode
    total_ratio = baseline_total / q2fp8_total

    print(f"{'Total Time (ms)':<30} {baseline_total:<20.2f} {q2fp8_total:<20.2f} {total_ratio:<15.3f}x")
    print(f"{'Memory (MB)':<30} {baseline_results['memory_mb']:<20.2f} {q2fp8_results['memory_mb']:<20.2f} {q2fp8_results['memory_mb']/baseline_results['memory_mb']:<15.3f}x")

    # Summary
    print("\n" + "=" * 100)
    if decode_ratio > 1.0:
        print(f"SUMMARY: Q2FP8 is {decode_ratio:.2f}x FASTER in decode ({1000/q2fp8_per_token:.2f} vs {1000/baseline_per_token:.2f} tok/s)")
    else:
        print(f"SUMMARY: Q2FP8 is {1/decode_ratio:.2f}x SLOWER in decode ({1000/q2fp8_per_token:.2f} vs {1000/baseline_per_token:.2f} tok/s)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefill and decode phases separately")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to the model")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[512, 2048, 8192, 32768],
                        help="List of prompt lengths to test")
    parser.add_argument("--decode_lengths", type=int, nargs="+", default=[1, 32, 128, 512],
                        help="List of decode lengths to test")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--output", type=str, default="prefill_decode_benchmark.json",
                        help="Output JSON file")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline benchmark")
    parser.add_argument("--skip_q2fp8", action="store_true", help="Skip Q2FP8 benchmark")
    parser.add_argument("--max_decode_tokens", type=int, default=4096, help="Max decode tokens for CUDA Graph buffer")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 100)
    print("PREFILL + DECODE BENCHMARK")
    print("=" * 100)
    print(f"Model: {args.model_path}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Decode lengths: {args.decode_lengths}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Device: {device}")
    print("=" * 100)

    # Load tokenizer for prompt generation
    temp_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 10

    all_results = []

    # Load models once before the loop
    q2fp8_model = None
    q2fp8_tokenizer = None
    Q2FP8SymCache = None
    max_decode_tokens = None
    baseline_model = None
    baseline_tokenizer = None

    if not args.skip_q2fp8:
        print("\n--- Loading Q2FP8 Model ---")
        q2fp8_model, q2fp8_tokenizer, Q2FP8SymCache, max_decode_tokens = load_q2fp8_model(
            args.model_path, args.device, dtype, max_decode_tokens=args.max_decode_tokens
        )

        # Simple warmup: let CUDA Graph capture naturally during first decode
        print("Warming up Q2FP8 model...")
        warmup_inputs = q2fp8_tokenizer("Hello", return_tensors="pt").to(device)
        warmup_cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)
        with torch.no_grad():
            _ = q2fp8_model.generate(
                **warmup_inputs,
                max_new_tokens=256,
                past_key_values=warmup_cache,
                use_cache=True,
                pad_token_id=q2fp8_tokenizer.pad_token_id,
                eos_token_id=q2fp8_tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        print("Warmup complete!")

    if not args.skip_baseline:
        print("\n--- Loading Baseline Model ---")
        baseline_model, baseline_tokenizer = load_baseline_model(args.model_path, device, dtype)

        # Warmup
        warmup_inputs = baseline_tokenizer("Hello", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = baseline_model.generate(**warmup_inputs, max_new_tokens=5)
        torch.cuda.empty_cache()
        print("Baseline warmup complete!")

    for prompt_len in args.prompt_lengths:
        for decode_len in args.decode_lengths:
            print(f"\n{'='*100}")
            print(f"Testing: Prompt Length = {prompt_len}, Decode Length = {decode_len}")
            print(f"{'='*100}")

            # Generate prompt
            prompt = generate_repeated_prompt(base_prompt, prompt_len, temp_tokenizer)

            config = {
                "prompt_length": prompt_len,
                "decode_length": decode_len,
                "num_runs": args.num_runs,
            }

            baseline_results = None
            q2fp8_results = None

            # Benchmark Q2FP8
            if not args.skip_q2fp8:
                print("\n--- Benchmarking Q2FP8 ---")
                q2fp8_results = benchmark_q2fp8(
                    q2fp8_model, q2fp8_tokenizer, Q2FP8SymCache, prompt, decode_len, device,
                    max_decode_tokens=max_decode_tokens,
                    num_runs=args.num_runs
                )
                torch.cuda.empty_cache()

            # Benchmark Baseline
            if not args.skip_baseline:
                print("\n--- Benchmarking Baseline ---")
                baseline_results = benchmark_baseline(
                    baseline_model, baseline_tokenizer, prompt, decode_len, device, args.num_runs
                )
                torch.cuda.empty_cache()

            # Print comparison
            if baseline_results and q2fp8_results:
                print_comparison(baseline_results, q2fp8_results, config)

            # Save results
            result_entry = {
                "config": config,
                "baseline": baseline_results,
                "q2fp8": q2fp8_results,
            }
            all_results.append(result_entry)

    # Save all results to JSON
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print(f"All results saved to: {output_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

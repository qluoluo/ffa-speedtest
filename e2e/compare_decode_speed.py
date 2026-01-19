#!/usr/bin/env python3
"""
对比 Q2FP8-Unified 与 Baseline 的 Decode 速度
"""

import sys
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "shared"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Import baseline utilities
from benchmark_utils import (
    BenchmarkResult,
    reset_memory_stats,
    measure_peak_memory,
    print_result,
    save_results,
    Timer,
)
from test_prompts import get_test_prompts, generate_repeated_prompt


DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"


def load_baseline_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load vanilla HuggingFace model with Flash Attention 2."""
    print(f"Loading baseline model (Flash Attention 2) from {model_path}...")
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


def load_ffa_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    use_ffa_decode: bool = True,
    delta: float = 5.0,
    block_size: int = 128,
    k_bits: int = 2,
    use_fp8_residual: bool = True,
    use_cudagraph: bool = False,
):
    """Load FFA-Q2FP8-Unified model."""
    # Import FFA model components
    sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified" / "ffa_model"))

    # Compatibility patch for transformers 4.45.2
    import transformers.integrations
    if not hasattr(transformers.integrations, 'use_kernel_forward_from_hub'):
        sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified"))
        from compat_patch import use_kernel_forward_from_hub
        transformers.integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub

    from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
    from q2fp8_cache import Q2FP8SymCache

    print(f"Loading FFA-Q2FP8-Unified model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": use_ffa_decode,
        "delta": delta,
        "BS": block_size,
        "SBS": block_size,
        "use_fp8_residual": use_fp8_residual,
        "k_bits": k_bits,
        "use_cudagraph": use_cudagraph,
    }

    model = FFALlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="flash_attention_2",
    )
    model.eval()

    return model, tokenizer, config, Q2FP8SymCache


def benchmark_baseline_detailed(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Run baseline benchmark with manual token-by-token generation for precise timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    prefill_times = []
    decode_times = []
    total_times = []
    generated_tokens_list = []

    config = {
        "attn_implementation": "flash_attention_2",
        "detailed": True,
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

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

                # Don't stop at EOS - generate exactly max_new_tokens for fair comparison

        decode_timer.stop()
        decode_times.append(decode_timer.elapsed_ms)
        total_times.append(prefill_timer.elapsed_ms + decode_timer.elapsed_ms)
        generated_tokens_list.append(len(generated_ids))

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
        method_name="Flash Attention 2 (baseline)",
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


def benchmark_ffa_detailed(
    model,
    tokenizer,
    config,
    Q2FP8SymCache,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    block_size: int = 128,
    k_bits: int = 2,
    use_fp8_residual: bool = True,
    use_cudagraph: bool = False,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Run FFA-Q2FP8-Unified benchmark with manual token-by-token generation for precise timing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    prefill_times = []
    decode_times = []
    total_times = []
    generated_tokens_list = []

    benchmark_config = {
        "block_size": block_size,
        "k_bits": k_bits,
        "use_fp8_residual": use_fp8_residual,
        "use_ffa_decode": config.attn_settings.get("use_ffa_decode", True),
        "delta": config.attn_settings.get("delta", 5.0),
        "use_cudagraph": use_cudagraph,
        "detailed": True,
    }

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Create fresh cache
        cache = Q2FP8SymCache(BS=block_size, use_fp8_residual=use_fp8_residual, k_bits=k_bits)

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
                past_key_values=cache,
                use_cache=True,
            )

        prefill_timer.stop()
        prefill_times.append(prefill_timer.elapsed_ms)

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token]

        # Decode phase with CUDA Graph support
        decode_timer = Timer()
        decode_timer.start()

        graph = None
        static_input_ids = None
        static_attention_mask = None
        static_outputs = None

        if use_cudagraph and device.type == "cuda" and run_idx == 0:
            # Capture CUDA Graph on first run
            print("  [CUDA Graph] Capturing decode graph...")

            # Warmup for graph capture (without attention_mask to avoid CUDA Graph issues)
            for _ in range(3):
                with torch.no_grad():
                    _ = model(
                        input_ids=next_token,
                        attention_mask=None,  # Don't use attention_mask during graph capture
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            # Create static tensors for graph
            static_input_ids = torch.zeros_like(next_token)

            # Synchronize before capture
            torch.cuda.synchronize()

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_outputs = model(
                    input_ids=static_input_ids,
                    attention_mask=None,  # Don't use attention_mask in graph
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            print("  [CUDA Graph] Graph captured successfully!")

        with torch.no_grad():
            for step_idx in range(max_new_tokens - 1):
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                    ], dim=1)

                if use_cudagraph and graph is not None and device.type == "cuda":
                    # Use CUDA Graph replay (without attention_mask)
                    static_input_ids.copy_(next_token)
                    graph.replay()
                    outputs = static_outputs
                else:
                    # Standard forward pass
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token)

                # Don't stop at EOS - generate exactly max_new_tokens for fair comparison

        decode_timer.stop()
        decode_times.append(decode_timer.elapsed_ms)
        total_times.append(prefill_timer.elapsed_ms + decode_timer.elapsed_ms)
        generated_tokens_list.append(len(generated_ids))

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
        method_name=f"FFA-Q2FP8-Unified (k_bits={k_bits}, cudagraph={use_cudagraph})",
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_time_ms=avg_prefill_time,
        decode_time_ms=avg_decode_time,
        total_time_ms=avg_total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        total_throughput=total_throughput,
        memory_peak_mb=memory_peak,
        config=benchmark_config,
    )


def print_comparison(baseline_result: BenchmarkResult, ffa_result: BenchmarkResult):
    """Print comparison between baseline and FFA results."""
    print("\n" + "=" * 80)
    print("DECODE SPEED COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Q2FP8-Unified':<20} {'Speedup':<15}")
    print("-" * 85)

    # Decode time
    decode_speedup = baseline_result.decode_time_ms / ffa_result.decode_time_ms
    print(f"{'Decode Time (ms)':<30} {baseline_result.decode_time_ms:<20.2f} {ffa_result.decode_time_ms:<20.2f} {decode_speedup:<15.3f}x")

    # Decode throughput
    throughput_speedup = ffa_result.decode_throughput / baseline_result.decode_throughput
    print(f"{'Decode Throughput (tok/s)':<30} {baseline_result.decode_throughput:<20.2f} {ffa_result.decode_throughput:<20.2f} {throughput_speedup:<15.3f}x")

    # Prefill time
    prefill_speedup = baseline_result.prefill_time_ms / ffa_result.prefill_time_ms
    print(f"{'Prefill Time (ms)':<30} {baseline_result.prefill_time_ms:<20.2f} {ffa_result.prefill_time_ms:<20.2f} {prefill_speedup:<15.3f}x")

    # Total time
    total_speedup = baseline_result.total_time_ms / ffa_result.total_time_ms
    print(f"{'Total Time (ms)':<30} {baseline_result.total_time_ms:<20.2f} {ffa_result.total_time_ms:<20.2f} {total_speedup:<15.3f}x")

    # Memory
    memory_ratio = ffa_result.memory_peak_mb / baseline_result.memory_peak_mb
    print(f"{'Peak Memory (MB)':<30} {baseline_result.memory_peak_mb:<20.2f} {ffa_result.memory_peak_mb:<20.2f} {memory_ratio:<15.3f}x")

    print("\n" + "=" * 80)
    print(f"Summary: Q2FP8-Unified achieves {decode_speedup:.2f}x decode speedup")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare Q2FP8-Unified vs Baseline decode speed")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to the model")
    parser.add_argument("--prompt_type", choices=["short", "medium", "long", "custom"], default="medium")
    parser.add_argument("--prompt_tokens", type=int, default=None, help="Target prompt length in tokens")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for page-wise quantization")
    parser.add_argument("--k_bits", type=int, default=2, choices=[2, 4], help="Quantization bits")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta threshold")
    parser.add_argument("--use_fp8_residual", action="store_true", default=True, help="Use FP8 residual")
    parser.add_argument("--no_fp8_residual", action="store_false", dest="use_fp8_residual")
    parser.add_argument("--no_ffa_decode", action="store_true", help="Disable FFA decode path")
    parser.add_argument("--use_cudagraph", action="store_true", help="Enable CUDA Graph for decode acceleration")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 80)
    print("Q2FP8-Unified vs Baseline Decode Speed Comparison")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Device: {device}")

    # Get prompt
    prompts = get_test_prompts()
    if args.prompt_type == "custom" and args.prompt_tokens:
        # Use baseline tokenizer for prompt generation
        temp_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        prompt = generate_repeated_prompt(prompts["medium"], args.prompt_tokens, temp_tokenizer)
    else:
        prompt = prompts[args.prompt_type]

    # ========== Benchmark Baseline ==========
    print("\n" + "=" * 80)
    print("STEP 1: Benchmarking Baseline (Flash Attention 2)")
    print("=" * 80)

    baseline_model, baseline_tokenizer = load_baseline_model(args.model_path, device, dtype)

    # Warmup
    print("\nWarming up baseline model...")
    warmup_text = "Hello world"
    warmup_inputs = baseline_tokenizer(warmup_text, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = baseline_model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False,
                                   pad_token_id=baseline_tokenizer.pad_token_id)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nRunning baseline benchmark...")
    baseline_result = benchmark_baseline_detailed(
        baseline_model, baseline_tokenizer, prompt, args.max_new_tokens, device, num_runs=args.num_runs
    )

    print_result(baseline_result)

    # Clean up baseline model
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== Benchmark FFA ==========
    print("\n" + "=" * 80)
    print("STEP 2: Benchmarking FFA-Q2FP8-Unified")
    print("=" * 80)

    ffa_model, ffa_tokenizer, ffa_config, Q2FP8SymCache = load_ffa_model(
        args.model_path, device, dtype,
        use_ffa_decode=not args.no_ffa_decode,
        delta=args.delta,
        block_size=args.block_size,
        k_bits=args.k_bits,
        use_fp8_residual=args.use_fp8_residual,
        use_cudagraph=args.use_cudagraph,
    )

    print(f"\nFFA Config:")
    print(f"  Block size: {args.block_size}")
    print(f"  K bits: {args.k_bits}")
    print(f"  Delta: {args.delta}")
    print(f"  Use FP8 residual: {args.use_fp8_residual}")
    print(f"  Use FFA decode: {not args.no_ffa_decode}")
    print(f"  Use CUDA Graph: {args.use_cudagraph}")

    # Warmup
    print("\nWarming up FFA model...")
    warmup_inputs = ffa_tokenizer(warmup_text, return_tensors="pt").to(device)
    warmup_cache = Q2FP8SymCache(BS=args.block_size, use_fp8_residual=args.use_fp8_residual, k_bits=args.k_bits)
    with torch.no_grad():
        _ = ffa_model.generate(
            **warmup_inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=ffa_tokenizer.pad_token_id,
            past_key_values=warmup_cache,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nRunning FFA benchmark...")
    ffa_result = benchmark_ffa_detailed(
        ffa_model, ffa_tokenizer, ffa_config, Q2FP8SymCache, prompt, args.max_new_tokens, device,
        block_size=args.block_size, k_bits=args.k_bits,
        use_fp8_residual=args.use_fp8_residual, use_cudagraph=args.use_cudagraph, num_runs=args.num_runs
    )

    print_result(ffa_result)

    # ========== Print Comparison ==========
    print_comparison(baseline_result, ffa_result)

    # Save results
    if args.output:
        save_results([baseline_result, ffa_result], args.output)
    else:
        default_output = Path(__file__).parent / "decode_speed_comparison.json"
        save_results([baseline_result, ffa_result], str(default_output))
        print(f"\nResults saved to: {default_output}")


if __name__ == "__main__":
    main()

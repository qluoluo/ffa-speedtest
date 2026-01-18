#!/usr/bin/env python3
"""Shared utilities for end-to-end speed benchmarking."""

import time
import json
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    prompt_length: int
    generated_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    prefill_throughput: float  # tokens/s
    decode_throughput: float   # tokens/s
    total_throughput: float    # tokens/s
    memory_peak_mb: float
    config: dict

    def to_dict(self):
        return asdict(self)


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self):
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@contextmanager
def cuda_timer():
    """Context manager for timing CUDA operations."""
    timer = Timer()
    timer.start()
    yield timer
    timer.stop()


def measure_peak_memory():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def warmup_model(model, tokenizer, device, num_warmup=3):
    """Warmup the model with short sequences."""
    print(f"  Warming up model ({num_warmup} iterations)...")
    warmup_text = "Hello, this is a warmup sequence."
    inputs = tokenizer(warmup_text, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_benchmark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    method_name: str,
    config: dict,
    prepare_cache_fn: Optional[Callable] = None,
    cleanup_fn: Optional[Callable] = None,
    num_runs: int = 3,
) -> BenchmarkResult:
    """
    Run end-to-end benchmark for a model.

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        max_new_tokens: Number of tokens to generate
        device: Device to run on
        method_name: Name of the method being benchmarked
        config: Configuration dict for the method
        prepare_cache_fn: Optional function to prepare cache before generation
        cleanup_fn: Optional function to cleanup after generation
        num_runs: Number of runs to average

    Returns:
        BenchmarkResult with timing statistics
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    prefill_times = []
    decode_times = []
    total_times = []

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Prepare cache if needed
        cache_kwargs = {}
        if prepare_cache_fn:
            cache_kwargs = prepare_cache_fn()

        # Total generation timing
        total_timer = Timer()
        total_timer.start()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
                **cache_kwargs,
            )

        total_timer.stop()

        # Cleanup if needed
        if cleanup_fn:
            cleanup_fn()

        total_times.append(total_timer.elapsed_ms)

        # Note: For precise prefill/decode separation, we'd need model-level hooks
        # Here we estimate based on token count ratio
        generated_tokens = outputs.shape[1] - prompt_length

    # Calculate average times
    avg_total_time = sum(total_times) / len(total_times)

    # Estimate prefill vs decode time based on typical ratios
    # In practice, prefill is much faster per token than decode
    # A rough estimate: prefill takes ~10-20% of time for long prompts
    estimated_prefill_ratio = min(0.3, prompt_length / (prompt_length + generated_tokens * 10))
    prefill_time = avg_total_time * estimated_prefill_ratio
    decode_time = avg_total_time - prefill_time

    # Calculate throughput
    prefill_throughput = prompt_length / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_throughput = generated_tokens / (decode_time / 1000) if decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    return BenchmarkResult(
        method_name=method_name,
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


def run_detailed_benchmark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    method_name: str,
    config: dict,
    prepare_cache_fn: Optional[Callable] = None,
    cleanup_fn: Optional[Callable] = None,
    num_runs: int = 3,
) -> BenchmarkResult:
    """
    Run detailed benchmark with separate prefill and decode measurements.
    Uses manual token-by-token generation for accurate timing.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    prefill_times = []
    decode_times = []
    total_times = []

    for run_idx in range(num_runs):
        reset_memory_stats()

        # Prepare cache if needed
        past_key_values = None
        if prepare_cache_fn:
            cache_result = prepare_cache_fn()
            past_key_values = cache_result.get("past_key_values", None)

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
                past_key_values=past_key_values,
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
                if next_token.item() == (tokenizer.eos_token_id or -1):
                    break

        decode_timer.stop()
        decode_times.append(decode_timer.elapsed_ms)
        total_times.append(prefill_timer.elapsed_ms + decode_timer.elapsed_ms)

        # Cleanup if needed
        if cleanup_fn:
            cleanup_fn()

    # Calculate averages
    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_decode_time = sum(decode_times) / len(decode_times)
    avg_total_time = sum(total_times) / len(total_times)
    generated_tokens = len(generated_ids)

    # Calculate throughput
    prefill_throughput = prompt_length / (avg_prefill_time / 1000) if avg_prefill_time > 0 else 0
    decode_throughput = generated_tokens / (avg_decode_time / 1000) if avg_decode_time > 0 else 0
    total_throughput = (prompt_length + generated_tokens) / (avg_total_time / 1000)

    memory_peak = measure_peak_memory()

    return BenchmarkResult(
        method_name=method_name,
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


def print_result(result: BenchmarkResult):
    """Pretty print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"Method: {result.method_name}")
    print(f"{'='*60}")
    print(f"Prompt length:       {result.prompt_length} tokens")
    print(f"Generated tokens:    {result.generated_tokens} tokens")
    print(f"{'─'*60}")
    print(f"Prefill time:        {result.prefill_time_ms:.2f} ms")
    print(f"Decode time:         {result.decode_time_ms:.2f} ms")
    print(f"Total time:          {result.total_time_ms:.2f} ms")
    print(f"{'─'*60}")
    print(f"Prefill throughput:  {result.prefill_throughput:.2f} tokens/s")
    print(f"Decode throughput:   {result.decode_throughput:.2f} tokens/s")
    print(f"Total throughput:    {result.total_throughput:.2f} tokens/s")
    print(f"{'─'*60}")
    print(f"Peak memory:         {result.memory_peak_mb:.2f} MB")
    print(f"Config:              {result.config}")
    print(f"{'='*60}\n")


def save_results(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to JSON file."""
    data = [r.to_dict() for r in results]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")


def compare_results(results: List[BenchmarkResult]):
    """Compare multiple benchmark results."""
    if len(results) < 2:
        print("Need at least 2 results to compare")
        return

    baseline = results[0]

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline: {baseline.method_name}")
    print(f"{'─'*80}")

    headers = ["Method", "Total Time", "Speedup", "Decode Tok/s", "Memory"]
    print(f"{headers[0]:<25} {headers[1]:<15} {headers[2]:<10} {headers[3]:<15} {headers[4]:<10}")
    print(f"{'─'*80}")

    for result in results:
        speedup = baseline.total_time_ms / result.total_time_ms if result.total_time_ms > 0 else 0
        print(f"{result.method_name:<25} "
              f"{result.total_time_ms:>10.2f} ms  "
              f"{speedup:>6.2f}x    "
              f"{result.decode_throughput:>10.2f}    "
              f"{result.memory_peak_mb:>8.1f} MB")

    print(f"{'='*80}\n")

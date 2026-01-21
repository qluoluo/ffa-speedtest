"""
Benchmark comparison script for FFA-Q2FP8-Unified vs Flash Attention.

This script compares the performance of:
1. Custom FFA method (use_ffa_decode=True)
2. Flash Attention baseline (use_ffa_decode=False)

For given prefill and decode token counts, it measures:
- Prefill latency
- Decode latency and throughput (tokens/sec)
- Total generation time (prefill + decode)

Results are saved to output/<timestamp>/ directory with plots.

Usage:
    python benchmark_comparison.py --model_path <path> --prefill_len 16384 --decode_len 256
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Add paths for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import torch
from transformers import AutoTokenizer, AutoConfig

from ffa_model.modeling_llama import LlamaForCausalLM
from ffa_model.q2fp8_cache import Q2FP8SymCache


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _get_next_token(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, 1, V] -> next_token: [B, 1]
    next_token = logits.argmax(dim=-1)
    if next_token.dim() == 1:
        next_token = next_token.unsqueeze(-1)
    return next_token


def _run_prefill(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Q2FP8SymCache,
    device: str,
) -> Tuple[torch.Tensor, float]:
    _sync_if_cuda(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=cache,
            logits_to_keep=1,
        )
    _sync_if_cuda(device)
    prefill_time = time.perf_counter() - start_time
    return outputs.logits, prefill_time


def _run_decode_steps(
    model: LlamaForCausalLM,
    next_token: torch.Tensor,
    num_steps: int,
    cache: Q2FP8SymCache,
    device: str,
) -> torch.Tensor:
    for _ in range(num_steps):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                use_cache=True,
                past_key_values=cache,
                logits_to_keep=1,
            )
        next_token = _get_next_token(outputs.logits)
    return next_token


def _save_plots(summary: Dict[str, Any], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    results = [r for r in summary["results"] if r.get("success")]
    if not results:
        return

    methods = [r["method"] for r in results]
    prefill_times = [r.get("prefill_time_sec", 0.0) for r in results]
    decode_times = [r.get("decode_time_sec", 0.0) for r in results]
    decode_tps = [r.get("decode_throughput_tokens_per_sec", 0.0) for r in results]

    try:
        import numpy as np
    except ImportError:
        print("numpy not available; skipping plots.")
        return

    x = np.arange(len(methods))
    width = 0.35

    # Prefill/Decode timing bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, prefill_times, width, label="Prefill")
    plt.bar(x + width / 2, decode_times, width, label="Decode")
    plt.xticks(x, methods)
    plt.ylabel("Time (s)")
    plt.title("Prefill vs Decode Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timings_prefill_decode.png"))
    plt.close()

    # Decode throughput bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(x, decode_tps, width=0.5)
    plt.xticks(x, methods)
    plt.ylabel("Tokens/sec")
    plt.title("Decode Throughput")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_decode.png"))
    plt.close()


def create_test_input(tokenizer, seq_len: int) -> str:
    """Create a test prompt with approximately seq_len tokens."""
    base_prompt = "You are an intelligent AI assistant. Please summarize the following text:\n\n"

    filler_unit = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test sentence to fill up the context window. "
        "Machine learning models need large contexts to test their capabilities. "
    )

    filler_tokens = len(tokenizer.encode(filler_unit, add_special_tokens=False))
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=False))

    target_filler_tokens = seq_len - base_tokens - 100
    num_units = max(1, target_filler_tokens // filler_tokens)

    filler_text = filler_unit * num_units
    full_prompt = base_prompt + filler_text + "\n\nSummary:"

    return full_prompt


def run_single_benchmark(
    model_path: str,
    tokenizer: Any,
    prefill_len: int,
    decode_len: int,
    use_ffa_decode: bool,
    delta: float = 5.0,
    BS: int = 128,
    k_bits: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
    warmup: bool = True,
    align_to_bs: bool = True,
    warmup_decode_tokens: int = 4,
    debug_stats: bool = False,
) -> Dict[str, Any]:
    """
    Run a single benchmark configuration.

    Returns:
        Dictionary with timing results and metadata
    """
    method_name = "FFA-Q2FP8-Unified" if use_ffa_decode else "FlashAttention"
    print(f"\n{'='*70}")
    print(f"Running benchmark: {method_name}")
    print(f"{'='*70}")
    print(f"Prefill length: {prefill_len}, Decode length: {decode_len}")
    print(f"use_ffa_decode: {use_ffa_decode}")
    if use_ffa_decode:
        print(f"delta: {delta}, BS: {BS}, k_bits: {k_bits}")

    # Load model config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": use_ffa_decode,
        "delta": delta,
        "BS": BS,
        "k_bits": k_bits,
        "use_fp8_residual": True,
        "return_skip_ratio": False,
    }
    stats_ref = None
    if debug_stats:
        config.attn_settings["debug_stats"] = {}

    # Load model
    print(f"Loading model...")
    torch_dtype = getattr(torch, dtype)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {device}")

    # Create test input
    test_prompt = create_test_input(tokenizer, prefill_len)
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=prefill_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    actual_prefill_len = input_ids.shape[1]
    print(f"Actual prefill length: {actual_prefill_len} tokens")

    # Create cache - only use Q2FP8 cache for FFA decode
    if use_ffa_decode:
        cache = Q2FP8SymCache(BS=BS, use_fp8_residual=True, k_bits=k_bits)
    else:
        cache = None  # Use standard transformers cache for Flash Attention

    # Warmup + benchmark run
    try:
        if warmup:
            print("Running warmup...")
            if use_ffa_decode:
                warmup_cache = Q2FP8SymCache(BS=BS, use_fp8_residual=True, k_bits=k_bits)
            else:
                warmup_cache = None
            warmup_logits, _ = _run_prefill(model, input_ids, attention_mask, warmup_cache, device)
            warmup_next = _get_next_token(warmup_logits)
            _run_decode_steps(
                model,
                warmup_next,
                num_steps=min(4, decode_len),
                cache=warmup_cache,
                device=device,
            )
            if device == "cuda":
                torch.cuda.empty_cache()
            print("Warmup complete")

        # Actual benchmark run (prefill timed)
        print("Running benchmark...")
        logits, prefill_time = _run_prefill(model, input_ids, attention_mask, cache, device)
        next_token = _get_next_token(logits)

        # Optional alignment to BS boundary (not timed) - only for FFA decode
        align_tokens = 0
        align_time = 0.0
        if use_ffa_decode and align_to_bs and cache is not None and cache.get_current_len() > 0:
            align_tokens = (BS - cache.get_current_len()) % BS
        if align_tokens > 0:
            print(f"Aligning cache to BS boundary with {align_tokens} tokens (not timed)...")
            _sync_if_cuda(device)
            align_start = time.perf_counter()
            next_token = _run_decode_steps(
                model,
                next_token,
                num_steps=align_tokens,
                cache=cache,
                device=device,
            )
            _sync_if_cuda(device)
            align_time = time.perf_counter() - align_start

        # Warmup decode steps to trigger CUDAGraph capture (not timed)
        warmup_decode_tokens = max(0, int(warmup_decode_tokens))
        warmup_decode_time = 0.0
        if warmup_decode_tokens > 0:
            print(f"Running {warmup_decode_tokens} decode warmup steps (not timed)...")
            _sync_if_cuda(device)
            warmup_start = time.perf_counter()
            next_token = _run_decode_steps(
                model,
                next_token,
                num_steps=warmup_decode_tokens,
                cache=cache,
                device=device,
            )
            _sync_if_cuda(device)
            warmup_decode_time = time.perf_counter() - warmup_start

        # Timed decode
        print("Running timed decode...")
        _sync_if_cuda(device)
        decode_start = time.perf_counter()
        _ = _run_decode_steps(
            model,
            next_token,
            num_steps=decode_len,
            cache=cache,
            device=device,
        )
        _sync_if_cuda(device)
        decode_time = time.perf_counter() - decode_start

        # Metrics
        total_time = prefill_time + decode_time
        decode_throughput = decode_len / decode_time if decode_time > 0 else 0.0

        result = {
            "method": method_name,
            "use_ffa_decode": use_ffa_decode,
            "success": True,
            "prefill_len_target": prefill_len,
            "prefill_len_actual": actual_prefill_len,
            "decode_len_target": decode_len,
            "decode_len_actual": decode_len,
            "prefill_time_sec": prefill_time,
            "decode_time_sec": decode_time,
            "total_time_sec": total_time,
            "benchmark_wall_time_sec": prefill_time + align_time + warmup_decode_time + decode_time,
            "align_tokens": align_tokens,
            "align_time_sec": align_time,
            "warmup_decode_tokens": warmup_decode_tokens,
            "warmup_decode_time_sec": warmup_decode_time,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "config": {
                "delta": delta if use_ffa_decode else None,
                "BS": BS,
                "k_bits": k_bits,
                "dtype": dtype,
                "align_to_bs": align_to_bs,
            }
        }
        if debug_stats:
            stats_ref = model.config.attn_settings.get("debug_stats")
            if stats_ref is not None:
                result["debug_stats"] = stats_ref

        print(f"\n{'='*70}")
        print(f"Benchmark SUCCESSFUL: {method_name}")
        print(f"{'='*70}")
        print(f"Prefill: {actual_prefill_len} tokens")
        print(f"Decode: {decode_len} tokens")
        print(f"Prefill time: {prefill_time:.3f}s")
        print(f"Decode time: {decode_time:.3f}s")
        print(f"Decode throughput: {decode_throughput:.2f} tokens/sec")
        print(f"{'='*70}")

        # Clean up
        del model
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"Benchmark FAILED: {method_name}")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # Clean up
        try:
            del model
            del cache
            if device == "cuda":
                torch.cuda.empty_cache()
        except:
            pass

        return {
            "method": method_name,
            "use_ffa_decode": use_ffa_decode,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_comparison_benchmark(
    model_path: str,
    prefill_len: int = 16384,
    decode_len: int = 256,
    delta: float = 5.0,
    BS: int = 128,
    k_bits: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
    output_dir: str = "output",
    align_to_bs: bool = True,
    warmup_decode_tokens: int = 4,
    debug_stats: bool = False,
):
    """
    Run comparison benchmark between FFA and Flash Attention.
    """
    print("="*70)
    print("FFA-Q2FP8-Unified vs Flash Attention Benchmark Comparison")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Prefill length: {prefill_len}")
    print(f"Decode length: {decode_len}")
    print(f"Device: {device}, dtype: {dtype}")
    print("="*70)

    # Load tokenizer once
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Run benchmarks
    results = []

    # 1. Flash Attention baseline
    print("\n" + "="*70)
    print("BENCHMARK 1/2: Flash Attention (Baseline)")
    print("="*70)
    flash_result = run_single_benchmark(
        model_path=model_path,
        tokenizer=tokenizer,
        prefill_len=prefill_len,
        decode_len=decode_len,
        use_ffa_decode=False,
        delta=delta,
        BS=BS,
        k_bits=k_bits,
        device=device,
        dtype=dtype,
        warmup=True,
        align_to_bs=align_to_bs,
        warmup_decode_tokens=warmup_decode_tokens,
        debug_stats=debug_stats,
    )
    results.append(flash_result)

    # 2. FFA method
    print("\n" + "="*70)
    print("BENCHMARK 2/2: FFA-Q2FP8-Unified (Custom Method)")
    print("="*70)
    ffa_result = run_single_benchmark(
        model_path=model_path,
        tokenizer=tokenizer,
        prefill_len=prefill_len,
        decode_len=decode_len,
        use_ffa_decode=True,
        delta=delta,
        BS=BS,
        k_bits=k_bits,
        device=device,
        dtype=dtype,
        warmup=True,
        align_to_bs=align_to_bs,
        warmup_decode_tokens=warmup_decode_tokens,
        debug_stats=debug_stats,
    )
    results.append(ffa_result)

    # Generate summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "device": device,
        "dtype": dtype,
        "results": results,
    }

    # Print comparison
    if flash_result["success"] and ffa_result["success"]:
        flash_throughput = flash_result["decode_throughput_tokens_per_sec"]
        ffa_throughput = ffa_result["decode_throughput_tokens_per_sec"]
        speedup = ffa_throughput / flash_throughput if flash_throughput > 0 else 0

        print(f"\nFlash Attention:")
        print(f"  Total time: {flash_result['total_time_sec']:.3f}s")
        print(f"  Decode throughput: {flash_throughput:.2f} tokens/sec")

        print(f"\nFFA-Q2FP8-Unified:")
        print(f"  Total time: {ffa_result['total_time_sec']:.3f}s")
        print(f"  Decode throughput: {ffa_throughput:.2f} tokens/sec")

        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time reduction: {(1 - 1/speedup)*100:.1f}%" if speedup > 0 else "N/A")

        summary["comparison"] = {
            "flash_throughput": flash_throughput,
            "ffa_throughput": ffa_throughput,
            "speedup": speedup,
            "time_reduction_percent": (1 - 1/speedup)*100 if speedup > 0 else None,
        }
    else:
        print("\nComparison not available due to benchmark failures.")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        output_dir,
        f"benchmark_comparison_{prefill_len}p_{decode_len}d_{timestamp_str}",
    )
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, "benchmark_comparison.json")

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    _save_plots(summary, run_dir)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"Plots saved to: {run_dir}")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison: FFA-Q2FP8-Unified vs Flash Attention"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B",
        help="Path to the Llama model",
    )
    parser.add_argument(
        "--prefill_len",
        type=int,
        default=16384,
        help="Prefill sequence length (default: 16384)",
    )
    parser.add_argument(
        "--decode_len",
        type=int,
        default=256,
        help="Number of decode tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help="Threshold delta for FFA (default: 5.0)",
    )
    parser.add_argument(
        "--BS",
        type=int,
        default=128,
        help="Block size for quantization (default: 128)",
    )
    parser.add_argument(
        "--k_bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="Quantization bits (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--align_to_bs",
        action="store_true",
        help="Align cache to BS boundary before timed decode",
    )
    parser.add_argument(
        "--no_align_to_bs",
        action="store_false",
        dest="align_to_bs",
        help="Disable aligning cache to BS boundary before timed decode",
    )
    parser.set_defaults(align_to_bs=True)
    parser.add_argument(
        "--warmup_decode_tokens",
        type=int,
        default=4,
        help="Number of decode warmup tokens before timing (default: 4)",
    )
    parser.add_argument(
        "--debug_stats",
        action="store_true",
        help="Collect debug stats from attention path selection",
    )

    args = parser.parse_args()

    summary = run_comparison_benchmark(
        model_path=args.model_path,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        delta=args.delta,
        BS=args.BS,
        k_bits=args.k_bits,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        align_to_bs=args.align_to_bs,
        warmup_decode_tokens=args.warmup_decode_tokens,
        debug_stats=args.debug_stats,
    )

    # Exit with success if both benchmarks succeeded
    all_success = all(r.get("success", False) for r in summary["results"])
    exit(0 if all_success else 1)


if __name__ == "__main__":
    main()

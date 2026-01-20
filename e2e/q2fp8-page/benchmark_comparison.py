"""
Benchmark comparison script for FFA-Q2FP8-Page vs Flash Attention.

This script compares the performance of:
1. Custom FFA method (use_ffa_decode=True)
2. Flash Attention baseline (use_ffa_decode=False)

For given prefill and decode token counts, it measures:
- Prefill latency
- Decode throughput (tokens/sec)
- Total generation time

Results are saved to output/ directory with timestamp.

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
) -> Dict[str, Any]:
    """
    Run a single benchmark configuration.

    Returns:
        Dictionary with timing results and metadata
    """
    method_name = "FFA-Q2FP8" if use_ffa_decode else "FlashAttention"
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

    # Create cache
    cache = Q2FP8SymCache(BS=BS, use_fp8_residual=True, k_bits=k_bits)

    # Warmup run
    if warmup:
        print("Running warmup...")
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(32, decode_len),
                min_new_tokens=1,
                do_sample=False,
                past_key_values=Q2FP8SymCache(BS=BS, use_fp8_residual=True, k_bits=k_bits),
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("Warmup complete")

    # Actual benchmark run
    print("Running benchmark...")
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=decode_len,
                min_new_tokens=decode_len,  # Force exact decode length
                do_sample=False,
                past_key_values=cache,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        generated_ids = outputs[0, actual_prefill_len:]
        actual_decode_len = len(generated_ids)

        # Estimate prefill vs decode time (rough approximation)
        # Assume first token is mostly prefill, rest is decode
        decode_throughput = actual_decode_len / total_time if total_time > 0 else 0

        result = {
            "method": method_name,
            "use_ffa_decode": use_ffa_decode,
            "success": True,
            "prefill_len_target": prefill_len,
            "prefill_len_actual": actual_prefill_len,
            "decode_len_target": decode_len,
            "decode_len_actual": actual_decode_len,
            "total_time_sec": total_time,
            "decode_throughput_tokens_per_sec": decode_throughput,
            "config": {
                "delta": delta if use_ffa_decode else None,
                "BS": BS,
                "k_bits": k_bits,
                "dtype": dtype,
            }
        }

        print(f"\n{'='*70}")
        print(f"Benchmark SUCCESSFUL: {method_name}")
        print(f"{'='*70}")
        print(f"Prefill: {actual_prefill_len} tokens")
        print(f"Decode: {actual_decode_len} tokens")
        print(f"Total time: {total_time:.3f}s")
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
):
    """
    Run comparison benchmark between FFA and Flash Attention.
    """
    print("="*70)
    print("FFA vs Flash Attention Benchmark Comparison")
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
    )
    results.append(flash_result)

    # 2. FFA method
    print("\n" + "="*70)
    print("BENCHMARK 2/2: FFA-Q2FP8 (Custom Method)")
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

        print(f"\nFFA-Q2FP8:")
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
    output_file = os.path.join(
        output_dir,
        f"benchmark_comparison_{prefill_len}p_{decode_len}d_{timestamp_str}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison: FFA-Q2FP8 vs Flash Attention"
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
    )

    # Exit with success if both benchmarks succeeded
    all_success = all(r.get("success", False) for r in summary["results"])
    exit(0 if all_success else 1)


if __name__ == "__main__":
    main()

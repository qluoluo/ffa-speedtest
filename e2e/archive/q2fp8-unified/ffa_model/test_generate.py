"""
Test script for FFA-Q2FP8-Sym LlamaForCausalLM generation with 16k context.

Usage:
    python test_generate.py --model_path <path_to_llama_model> [options]

Example:
    python test_generate.py --model_path /path/to/Llama-3.1-8B-Instruct --seq_len 16000
"""

import argparse
import os
import sys
import time

# Add paths for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import torch
from transformers import AutoTokenizer, AutoConfig

from modeling_llama import LlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache


def create_test_input(tokenizer, seq_len: int = 16000) -> str:
    """Create a test prompt with approximately seq_len tokens."""
    # Base prompt
    base_prompt = "You are an intelligent AI assistant. Please summarize the following text:\n\n"

    # Generate filler text to reach target length
    filler_unit = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test sentence to fill up the context window. "
        "Machine learning models need large contexts to test their capabilities. "
    )

    # Estimate tokens per filler unit
    filler_tokens = len(tokenizer.encode(filler_unit, add_special_tokens=False))
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=False))

    # Calculate how many filler units we need
    target_filler_tokens = seq_len - base_tokens - 100  # Leave room for generation
    num_units = max(1, target_filler_tokens // filler_tokens)

    # Build the full prompt
    filler_text = filler_unit * num_units
    full_prompt = base_prompt + filler_text + "\n\nSummary:"

    return full_prompt


def test_generate(
    model_path: str,
    seq_len: int = 16000,
    max_new_tokens: int = 128,
    use_ffa_decode: bool = True,
    delta: float = 5.0,
    BS: int = 128,
    k_bits: int = 2,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """
    Test generation with FFA-Q2FP8-Sym attention.

    Args:
        model_path: Path to the Llama model
        seq_len: Target input sequence length
        max_new_tokens: Number of tokens to generate
        use_ffa_decode: Whether to use FFA decode path
        delta: Threshold delta for FFA
        BS: Block size for quantization
        k_bits: Quantization bits (2 or 4)
        device: Device to run on
        dtype: Model dtype
    """
    print("=" * 60)
    print("FFA-Q2FP8-Sym Generation Test")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Target seq_len: {seq_len}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"use_ffa_decode: {use_ffa_decode}")
    print(f"delta: {delta}, BS: {BS}, k_bits: {k_bits}")
    print(f"device: {device}, dtype: {dtype}")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load model config and set attention settings
    print("\n[2/5] Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": use_ffa_decode,
        "delta": delta,
        "BS": BS,
        "k_bits": k_bits,
        "use_fp8_residual": True,
        "return_skip_ratio": False,
    }
    print(f"  Config loaded. Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    print(f"  Num layers: {config.num_hidden_layers}")

    # Load model
    print("\n[3/5] Loading model...")
    torch_dtype = getattr(torch, dtype)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {device}")

    # Create test input
    print("\n[4/5] Creating test input...")
    test_prompt = create_test_input(tokenizer, seq_len)
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    actual_seq_len = input_ids.shape[1]
    print(f"  Input sequence length: {actual_seq_len} tokens")

    # Create fresh cache
    print("\n[5/5] Running generation...")
    cache = Q2FP8SymCache(BS=BS, use_fp8_residual=True, k_bits=k_bits)

    # Warm up (optional, for more accurate timing)
    torch.cuda.synchronize() if device == "cuda" else None

    # Generate
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                past_key_values=cache,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed_time = time.time() - start_time

        # Decode output
        generated_ids = outputs[0, actual_seq_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_generated = len(generated_ids)

        print("\n" + "=" * 60)
        print("Generation SUCCESSFUL!")
        print("=" * 60)
        print(f"Input length: {actual_seq_len} tokens")
        print(f"Generated tokens: {num_generated}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Tokens/sec (generation): {num_generated / elapsed_time:.2f}")
        print(f"\nGenerated text (first 200 chars):")
        print("-" * 40)
        print(generated_text[:200])
        print("-" * 40)

        # Cache statistics
        if hasattr(cache, 'layers') and len(cache.layers) > 0:
            layer0 = cache.layers[0]
            print(f"\nCache statistics (layer 0):")
            print(f"  Quantized length: {layer0.get_quantized_len()}")
            print(f"  Current length: {layer0.get_current_len()}")
            print(f"  Total seq length: {layer0.get_seq_length()}")
            print(f"  Num full blocks: {layer0.num_full_blocks}")
            if layer0.k_q is not None:
                print(f"  k_q shape: {layer0.k_q.shape}")
                print(f"  k_scale shape: {layer0.k_scale.shape}")
                print(f"  k_residual shape: {layer0.k_residual.shape}")
            if layer0.value is not None:
                print(f"  value shape: {layer0.value.shape}")

        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Generation FAILED!")
        print("=" * 60)
        print(f"Error after {elapsed_time:.2f}s:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FFA-Q2FP8-Sym generation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Llama model",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=16000,
        help="Target input sequence length (default: 16000)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--use_ffa_decode",
        action="store_true",
        default=True,
        help="Use FFA decode path (default: True)",
    )
    parser.add_argument(
        "--no_ffa_decode",
        action="store_true",
        help="Disable FFA decode (use standard attention)",
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

    args = parser.parse_args()

    # Handle --no_ffa_decode flag
    use_ffa = args.use_ffa_decode and not args.no_ffa_decode

    success = test_generate(
        model_path=args.model_path,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        use_ffa_decode=use_ffa,
        delta=args.delta,
        BS=args.BS,
        k_bits=args.k_bits,
        device=args.device,
        dtype=args.dtype,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

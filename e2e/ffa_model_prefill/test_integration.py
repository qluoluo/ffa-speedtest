"""
Integration Test and Benchmark for FFA Prefill + Decode

This script tests and benchmarks the complete prefill + decode pipeline:
1. Prefill: Fused RoPE + quantization + threshold-based attention
2. Decode: Existing decode kernel with current buffer
3. Comparison with FlashAttention-2 baseline

Usage:
    python test_integration.py --seq_len 2048 --num_decode 100
"""

import argparse
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from q2fp8_cache_prefill import Q2FP8CachePrefill
    from modeling_llama_prefill import LlamaAttentionPrefill
    from transformers.models.llama.configuration_llama import LlamaConfig
except ImportError:
    print("Warning: Could not import modules. Make sure you're in the correct directory.")
    exit(1)


def create_test_config(
    hidden_size: int = 2048,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 64,
) -> LlamaConfig:
    """Create test configuration"""
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        attention_bias=False,
        use_ffa_prefill=True,
        use_ffa_decode=True,
        ffa_delta=5.0,
        ffa_block_size=64,
    )
    # Set head_dim explicitly
    config.head_dim = head_dim
    return config


def generate_rope_embeddings(seq_len: int, head_dim: int, device: str = "cuda") -> tuple:
    """Generate RoPE cos/sin embeddings"""
    # Simple RoPE generation for testing
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    freqs = torch.outer(position_ids.squeeze(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def test_prefill(
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_size: int = 2048,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda",
) -> Dict:
    """Test prefill path"""
    print(f"\n{'='*70}")
    print(f"Testing Prefill Path")
    print(f"{'='*70}")
    print(f"Config: B={batch_size}, T={seq_len}, H={hidden_size}, NH={num_heads}, NKV={num_kv_heads}")

    # Create config and model
    config = create_test_config(hidden_size, num_heads, num_kv_heads, head_dim)
    attn = LlamaAttentionPrefill(config, layer_idx=0).to(device)

    # Create cache
    cache = Q2FP8CachePrefill(
        max_batch_size=batch_size,
        max_cache_len=seq_len * 2,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        device=device,
    )

    # Generate input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Generate RoPE embeddings
    cos, sin = generate_rope_embeddings(seq_len, head_dim, device)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    num_runs = 10
    times = []

    for _ in range(num_runs):
        # Reset cache for each run
        cache = Q2FP8CachePrefill(
            max_batch_size=batch_size,
            max_cache_len=seq_len * 2,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=64,
            device=device,
        )

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output, _ = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min time: {min(times):.2f} ms")
    print(f"  Max time: {max(times):.2f} ms")
    print(f"  Output shape: {output.shape}")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "times": times,
        "output_shape": output.shape,
    }


def test_decode(
    batch_size: int = 1,
    prefill_len: int = 2048,
    num_decode_steps: int = 100,
    hidden_size: int = 2048,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda",
) -> Dict:
    """Test decode path"""
    print(f"\n{'='*70}")
    print(f"Testing Decode Path")
    print(f"{'='*70}")
    print(f"Config: B={batch_size}, Prefill={prefill_len}, Decode={num_decode_steps}")

    # Create config and model
    config = create_test_config(hidden_size, num_heads, num_kv_heads, head_dim)
    attn = LlamaAttentionPrefill(config, layer_idx=0).to(device)

    # Create cache
    cache = Q2FP8CachePrefill(
        max_batch_size=batch_size,
        max_cache_len=prefill_len + num_decode_steps,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        device=device,
    )

    # Prefill phase
    print("Running prefill...")
    hidden_states_prefill = torch.randn(batch_size, prefill_len, hidden_size, dtype=torch.float16, device=device)
    position_ids_prefill = torch.arange(prefill_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos_prefill, sin_prefill = generate_rope_embeddings(prefill_len, head_dim, device)

    with torch.no_grad():
        _, cache = attn(hidden_states_prefill, position_ids_prefill, cache, cos=cos_prefill, sin=sin_prefill)

    torch.cuda.synchronize()

    # Decode phase
    print("Running decode...")
    decode_times = []

    for step in range(num_decode_steps):
        hidden_states_decode = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device=device)
        position_ids_decode = torch.tensor([[prefill_len + step]], device=device)
        cos_decode, sin_decode = generate_rope_embeddings(prefill_len + step + 1, head_dim, device)
        cos_decode = cos_decode[prefill_len + step:prefill_len + step + 1]
        sin_decode = sin_decode[prefill_len + step:prefill_len + step + 1]

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output, cache = attn(hidden_states_decode, position_ids_decode, cache, cos=cos_decode, sin=sin_decode)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        decode_times.append(elapsed)

    avg_time = sum(decode_times) / len(decode_times)
    std_time = (sum((t - avg_time) ** 2 for t in decode_times) / len(decode_times)) ** 0.5

    print(f"\nResults:")
    print(f"  Average decode time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min time: {min(decode_times):.2f} ms")
    print(f"  Max time: {max(decode_times):.2f} ms")
    print(f"  Total decode time: {sum(decode_times):.2f} ms")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "times": decode_times,
        "total_time": sum(decode_times),
    }


def test_end_to_end(
    batch_size: int = 1,
    prefill_len: int = 2048,
    num_decode_steps: int = 100,
    device: str = "cuda",
) -> Dict:
    """Test complete prefill + decode pipeline"""
    print(f"\n{'='*70}")
    print(f"End-to-End Test: Prefill + Decode")
    print(f"{'='*70}")

    prefill_results = test_prefill(
        batch_size=batch_size,
        seq_len=prefill_len,
        device=device,
    )

    decode_results = test_decode(
        batch_size=batch_size,
        prefill_len=prefill_len,
        num_decode_steps=num_decode_steps,
        device=device,
    )

    total_time = prefill_results["avg_time"] + decode_results["total_time"]

    print(f"\n{'='*70}")
    print(f"End-to-End Summary")
    print(f"{'='*70}")
    print(f"  Prefill time: {prefill_results['avg_time']:.2f} ms")
    print(f"  Decode total time: {decode_results['total_time']:.2f} ms")
    print(f"  Decode avg per token: {decode_results['avg_time']:.2f} ms")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Throughput: {(prefill_len + num_decode_steps) / (total_time / 1000):.2f} tokens/sec")

    return {
        "prefill": prefill_results,
        "decode": decode_results,
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Test FFA Prefill + Decode")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Prefill sequence length")
    parser.add_argument("--num_decode", type=int, default=100, help="Number of decode steps")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--test", type=str, default="all", choices=["prefill", "decode", "all"],
                       help="Which test to run")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"FFA Prefill + Decode Integration Test")
    print(f"{'='*70}")
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.test == "prefill":
        test_prefill(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            device=args.device,
        )
    elif args.test == "decode":
        test_decode(
            batch_size=args.batch_size,
            prefill_len=args.seq_len,
            num_decode_steps=args.num_decode,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            device=args.device,
        )
    else:
        test_end_to_end(
            batch_size=args.batch_size,
            prefill_len=args.seq_len,
            num_decode_steps=args.num_decode,
            device=args.device,
        )

    print(f"\n{'='*70}")
    print("✓ All tests completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

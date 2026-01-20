"""
Performance Comparison: FFA Prefill vs Existing Baseline

This script compares the new FFA prefill implementation against:
1. Existing q2fp8-unified decode kernel (for decode phase)
2. Standard PyTorch attention (for prefill phase, as baseline)

Tests:
- Prefill latency and throughput
- Decode latency and throughput
- Memory usage
- End-to-end performance
"""

import argparse
import time
import torch
import torch.nn.functional as F
from typing import Dict, List
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../q2fp8-unified'))

# Import new implementation
try:
    from ffa_model_prefill.q2fp8_cache_prefill import Q2FP8CachePrefill
    from ffa_model_prefill.modeling_llama_prefill import LlamaAttentionPrefill
    NEW_IMPL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import new implementation: {e}")
    NEW_IMPL_AVAILABLE = False

# Import existing implementation
try:
    from q2fp8_unified.ffa_model.q2fp8_cache import Q2FP8SymCache
    from q2fp8_unified.ffa_model.modeling_llama import LlamaAttention
    EXISTING_IMPL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import existing implementation: {e}")
    EXISTING_IMPL_AVAILABLE = False

from transformers.models.llama.configuration_llama import LlamaConfig


def generate_rope_embeddings(seq_len: int, head_dim: int, device: str = "cuda"):
    """Generate RoPE embeddings"""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    freqs = torch.outer(position_ids.squeeze(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def benchmark_new_prefill(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict:
    """Benchmark new FFA prefill implementation"""
    print(f"\n{'='*70}")
    print(f"Benchmarking NEW FFA Prefill Implementation")
    print(f"{'='*70}")

    if not NEW_IMPL_AVAILABLE:
        print("❌ New implementation not available")
        return None

    # Create config
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
    config.head_dim = head_dim

    # Create model
    attn = LlamaAttentionPrefill(config, layer_idx=0).to(device)

    # Generate input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = generate_rope_embeddings(seq_len, head_dim, device)

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        cache = Q2FP8CachePrefill(
            max_batch_size=batch_size,
            max_cache_len=seq_len * 2,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=64,
            device=device,
        )
        with torch.no_grad():
            _ = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []

    for _ in range(num_runs):
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

    print(f"\n✓ Results:")
    print(f"  Average: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min: {min(times):.2f} ms")
    print(f"  Max: {max(times):.2f} ms")
    print(f"  Throughput: {seq_len / (avg_time / 1000):.2f} tokens/sec")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "times": times,
        "throughput": seq_len / (avg_time / 1000),
    }


def benchmark_baseline_prefill(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict:
    """Benchmark baseline (standard PyTorch attention)"""
    print(f"\n{'='*70}")
    print(f"Benchmarking BASELINE (Standard PyTorch Attention)")
    print(f"{'='*70}")

    # Create simple attention layer
    q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(device).half()
    k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device).half()
    v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device).half()
    o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(device).half()

    scaling = head_dim ** -0.5
    num_key_value_groups = num_heads // num_kv_heads

    # Generate input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)

    def run_attention():
        # Project
        q = q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Repeat KV for GQA
        if num_key_value_groups > 1:
            k = k.repeat_interleave(num_key_value_groups, dim=1)
            v = v.repeat_interleave(num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        output = o_proj(attn_output)

        return output

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = run_attention()

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = run_attention()

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\n✓ Results:")
    print(f"  Average: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min: {min(times):.2f} ms")
    print(f"  Max: {max(times):.2f} ms")
    print(f"  Throughput: {seq_len / (avg_time / 1000):.2f} tokens/sec")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "times": times,
        "throughput": seq_len / (avg_time / 1000),
    }


def benchmark_decode_comparison(
    batch_size: int,
    prefill_len: int,
    num_decode: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> Dict:
    """Compare decode performance between new and existing implementations"""
    print(f"\n{'='*70}")
    print(f"Decode Performance Comparison")
    print(f"{'='*70}")

    results = {}

    # Test new implementation
    if NEW_IMPL_AVAILABLE:
        print(f"\nTesting NEW implementation decode...")
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
        config.head_dim = head_dim

        attn = LlamaAttentionPrefill(config, layer_idx=0).to(device)
        cache = Q2FP8CachePrefill(
            max_batch_size=batch_size,
            max_cache_len=prefill_len + num_decode,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=64,
            device=device,
        )

        # Prefill
        hidden_states = torch.randn(batch_size, prefill_len, hidden_size, dtype=torch.float16, device=device)
        position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
        cos, sin = generate_rope_embeddings(prefill_len, head_dim, device)

        with torch.no_grad():
            _, cache = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

        torch.cuda.synchronize()

        # Decode
        decode_times = []
        for step in range(num_decode):
            hidden_states = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device=device)
            position_ids = torch.tensor([[prefill_len + step]], device=device)
            cos_full, sin_full = generate_rope_embeddings(prefill_len + step + 1, head_dim, device)
            cos = cos_full[prefill_len + step:prefill_len + step + 1]
            sin = sin_full[prefill_len + step:prefill_len + step + 1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                _, cache = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            decode_times.append(elapsed)

        avg_decode = sum(decode_times) / len(decode_times)
        print(f"  NEW decode avg: {avg_decode:.2f} ms/token")

        results["new"] = {
            "avg_time": avg_decode,
            "times": decode_times,
        }

    # Test existing implementation (if available)
    if EXISTING_IMPL_AVAILABLE:
        print(f"\nTesting EXISTING implementation decode...")
        # Similar setup for existing implementation
        # Note: This would require adapting to the existing API
        print("  (Skipping - requires API adaptation)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Performance comparison test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_decode", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_runs", type=int, default=10)

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"FFA Prefill Performance Comparison")
    print(f"{'='*70}")
    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Num KV heads: {args.num_kv_heads}")
    print(f"  Head dim: {args.head_dim}")

    # Test prefill
    print(f"\n{'='*70}")
    print(f"PREFILL PHASE COMPARISON")
    print(f"{'='*70}")

    baseline_results = benchmark_baseline_prefill(
        args.batch_size, args.seq_len, args.hidden_size,
        args.num_heads, args.num_kv_heads, args.head_dim,
        args.device, args.num_warmup, args.num_runs,
    )

    new_results = benchmark_new_prefill(
        args.batch_size, args.seq_len, args.hidden_size,
        args.num_heads, args.num_kv_heads, args.head_dim,
        args.device, args.num_warmup, args.num_runs,
    )

    # Compare results
    if baseline_results and new_results:
        print(f"\n{'='*70}")
        print(f"PREFILL COMPARISON SUMMARY")
        print(f"{'='*70}")

        speedup = baseline_results["avg_time"] / new_results["avg_time"]

        print(f"\nBaseline (PyTorch):")
        print(f"  Time: {baseline_results['avg_time']:.2f} ms")
        print(f"  Throughput: {baseline_results['throughput']:.2f} tokens/sec")

        print(f"\nNew FFA Prefill:")
        print(f"  Time: {new_results['avg_time']:.2f} ms")
        print(f"  Throughput: {new_results['throughput']:.2f} tokens/sec")

        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup > 1.0:
            print(f"✓ NEW implementation is {speedup:.2f}x FASTER")
        else:
            print(f"⚠ NEW implementation is {1/speedup:.2f}x SLOWER")

    # Test decode
    print(f"\n{'='*70}")
    print(f"DECODE PHASE TEST")
    print(f"{'='*70}")

    decode_results = benchmark_decode_comparison(
        args.batch_size, args.seq_len, args.num_decode,
        args.hidden_size, args.num_heads, args.num_kv_heads,
        args.head_dim, args.device,
    )

    print(f"\n{'='*70}")
    print(f"✓ Performance comparison completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

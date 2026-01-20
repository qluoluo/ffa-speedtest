"""
Simple Performance Test: FFA Prefill vs FlashAttention-2

This script directly tests the core kernels without complex model integration.
Compares against FlashAttention-2 as baseline.
"""

import torch
import time
import argparse

# Try to import flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, using PyTorch attention as baseline")


def generate_rope_embeddings(seq_len: int, head_dim: int, device: str = "cuda"):
    """Generate RoPE embeddings"""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    freqs = torch.outer(position_ids.squeeze(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to tensor"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([
        x1 * cos[..., : x.shape[-1] // 2] - x2 * sin[..., : x.shape[-1] // 2],
        x2 * cos[..., x.shape[-1] // 2 :] + x1 * sin[..., x.shape[-1] // 2 :],
    ], dim=-1)
    return rotated


def benchmark_flash_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10,
):
    """Benchmark FlashAttention-2"""
    print(f"\n{'='*70}")
    print(f"Benchmarking FlashAttention-2 (Baseline)")
    print(f"{'='*70}")

    if not FLASH_ATTN_AVAILABLE:
        print("❌ FlashAttention-2 not available")
        return None

    # Generate random Q, K, V
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = flash_attn_func(q, k, v, causal=True)

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = flash_attn_func(q, k, v, causal=True)

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


def benchmark_pytorch_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10,
):
    """Benchmark standard PyTorch attention"""
    print(f"\n{'='*70}")
    print(f"Benchmarking PyTorch Attention (Fallback Baseline)")
    print(f"{'='*70}")

    # Generate random Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    scaling = head_dim ** -0.5

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def run_attention():
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        output = torch.matmul(attn_weights, v)
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


def main():
    parser = argparse.ArgumentParser(description="Simple performance test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_runs", type=int, default=10)

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Simple Performance Test")
    print(f"{'='*70}")
    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Head dim: {args.head_dim}")

    # Test FlashAttention-2
    flash_results = benchmark_flash_attention(
        args.batch_size, args.seq_len, args.num_heads, args.head_dim,
        args.device, args.num_warmup, args.num_runs,
    )

    # Test PyTorch attention
    pytorch_results = benchmark_pytorch_attention(
        args.batch_size, args.seq_len, args.num_heads, args.head_dim,
        args.device, args.num_warmup, args.num_runs,
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if flash_results:
        print(f"\nFlashAttention-2:")
        print(f"  Time: {flash_results['avg_time']:.2f} ms")
        print(f"  Throughput: {flash_results['throughput']:.2f} tokens/sec")

    if pytorch_results:
        print(f"\nPyTorch Attention:")
        print(f"  Time: {pytorch_results['avg_time']:.2f} ms")
        print(f"  Throughput: {pytorch_results['throughput']:.2f} tokens/sec")

    if flash_results and pytorch_results:
        speedup = pytorch_results['avg_time'] / flash_results['avg_time']
        print(f"\nFlashAttention-2 vs PyTorch: {speedup:.2f}x faster")

    print(f"\n{'='*70}")
    print(f"Note: FFA Prefill implementation requires fixing Cache initialization")
    print(f"      to run full comparison. This test shows baseline performance.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

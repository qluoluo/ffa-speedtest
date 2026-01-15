"""
速度测试: FFA-Sample (Triton Sample4 FP16) vs PyTorch

支持两种模式:
1. 普通模式: 直接调用kernel
2. CUDAGraph模式: 使用CUDAGraph优化，减少kernel launch开销

不需要 Quest 环境，直接对比 Triton 稀疏注意力和 PyTorch 全注意力。
"""

import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# 添加路径
sys.path.insert(0, "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-flashinfer-sample")

from ffa_sample.kernels import sample_k_fp16, attn_forward_decode_sample4

# 尝试导入 CUDAGraph runner
try:
    from ffa_sample.kernels.triton_sample import CUDAGraphDecodeRunnerSample4Q2
    HAS_CUDAGRAPH = True
except ImportError:
    HAS_CUDAGRAPH = False


@torch.no_grad()
def benchmark_cuda_event(fn, iters: int = 50, warmup: int = 10) -> float:
    """使用 CUDA Event 进行精确计时 (与用户原始benchmark一致)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def benchmark_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> Dict[str, float]:
    """Benchmark standard PyTorch attention."""
    B, Tq, HQ, K = q.shape
    _, T, HKV, _ = k.shape
    V = v.shape[-1]

    G = HQ // HKV
    scale = 1.0 / (K ** 0.5)

    # GQA expansion
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, K)
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, V)
    q_2d = q.squeeze(1)  # [B, HQ, K]

    def run_fn():
        scores = torch.einsum("bhk,bthk->bht", q_2d, k_exp) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.einsum("bht,bthv->bhv", attn_weights, v_exp)

    ms = benchmark_cuda_event(run_fn, iters=num_iters, warmup=num_warmup)
    return {"total_ms": ms}


def benchmark_ffa_sample(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    delta: float = 5.0,
    max_kept_ratio: float = 0.2,
    num_warmup: int = 10,
    num_iters: int = 100,
    use_cudagraph: bool = False,
) -> Dict[str, float]:
    """Benchmark FFA-Sample (Triton Sample4 FP16)."""
    B, T, HKV, K = k.shape
    device = k.device
    dtype = k.dtype

    results = {}

    # Preprocessing: sample K
    k_sample = sample_k_fp16(k, BS=page_size)
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

    # 获取 skip_ratio
    _, skip_ratio = attn_forward_decode_sample4(
        q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
        k_full=k, v=v, BS=page_size, delta=delta,
        max_kept_ratio=max_kept_ratio, return_skip_ratio=True,
    )
    results["skip_ratio"] = skip_ratio

    if use_cudagraph and HAS_CUDAGRAPH:
        # 使用 CUDAGraph 模式
        runner = CUDAGraphDecodeRunnerSample4Q2(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            BS=page_size,
            delta=delta,
            max_kept_ratio=max_kept_ratio,
            warmup=2,
        )

        def run_fn():
            return runner.replay_only()

        ms = benchmark_cuda_event(run_fn, iters=num_iters, warmup=num_warmup)
        results["forward_ms"] = ms
        results["total_ms"] = ms
        results["mode"] = "cudagraph"
    else:
        # 普通模式
        def run_fn():
            return attn_forward_decode_sample4(
                q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
                k_full=k, v=v, BS=page_size, delta=delta,
                max_kept_ratio=max_kept_ratio,
            )

        ms = benchmark_cuda_event(run_fn, iters=num_iters, warmup=num_warmup)
        results["forward_ms"] = ms
        results["total_ms"] = ms
        results["mode"] = "normal"

    return results


def run_benchmark(
    seq_lengths: List[int],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    delta: float = 5.0,
    max_kept_ratio: float = 0.2,
    num_warmup: int = 100,
    num_iters: int = 500,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    use_cudagraph: bool = True,
):
    """Run benchmark."""

    mode_str = "CUDAGraph" if (use_cudagraph and HAS_CUDAGRAPH) else "Normal"

    print("=" * 100)
    print(f"Sparse Attention Benchmark: FFA-Sample (Triton) vs PyTorch Full Attention [{mode_str} Mode]")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  page_size: {page_size}, delta: {delta}, max_kept_ratio: {max_kept_ratio}")
    print(f"  dtype: {dtype}, warmup: {num_warmup}, iterations: {num_iters}")
    print(f"  CUDAGraph available: {HAS_CUDAGRAPH}, using: {use_cudagraph and HAS_CUDAGRAPH}")
    print()

    # Header
    print(f"{'SeqLen':>8} | {'Blocks':>6} | {'PyTorch(ms)':>11} | {'FFA(ms)':>10} | {'Skip Ratio':>10} | {'Speedup':>10}")
    print("-" * 75)

    for seq_len in seq_lengths:
        num_blocks = (seq_len + page_size - 1) // page_size

        # Create inputs
        q = torch.randn(1, 1, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

        # PyTorch
        try:
            pytorch_results = benchmark_pytorch_attention(q, k, v, num_warmup, num_iters)
            pytorch_ms = pytorch_results["total_ms"]
        except Exception as e:
            print(f"PyTorch failed: {e}")
            pytorch_ms = float('nan')

        # FFA-Sample
        try:
            ffa_results = benchmark_ffa_sample(
                q, k, v, page_size=page_size, delta=delta,
                max_kept_ratio=max_kept_ratio, num_warmup=num_warmup, num_iters=num_iters,
                use_cudagraph=use_cudagraph,
            )
            ffa_ms = ffa_results["total_ms"]
            skip_ratio = ffa_results.get("skip_ratio", 0)
        except Exception as e:
            print(f"FFA-Sample failed for seq_len={seq_len}: {e}")
            ffa_ms = float('nan')
            skip_ratio = 0

        speedup = pytorch_ms / ffa_ms if ffa_ms > 0 else float('nan')

        print(f"{seq_len:>8} | {num_blocks:>6} | {pytorch_ms:>11.3f} | {ffa_ms:>10.3f} | {skip_ratio:>10.1%} | {speedup:>10.2f}x")

        del q, k, v
        torch.cuda.empty_cache()

    print()


def run_detailed_benchmark(
    seq_len: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    delta: float = 5.0,
    max_kept_ratio: float = 0.2,
    num_warmup: int = 100,
    num_iters: int = 500,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    use_cudagraph: bool = True,
):
    """Run detailed benchmark with kernel timings."""
    mode_str = "CUDAGraph" if (use_cudagraph and HAS_CUDAGRAPH) else "Normal"

    print("=" * 80)
    print(f"Detailed Benchmark (seq_len={seq_len}) [{mode_str} Mode]")
    print("=" * 80)

    num_blocks = (seq_len + page_size - 1) // page_size

    q = torch.randn(1, 1, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    print(f"\nConfiguration:")
    print(f"  seq_len: {seq_len}, num_blocks: {num_blocks}")
    print(f"  num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  CUDAGraph available: {HAS_CUDAGRAPH}, using: {use_cudagraph and HAS_CUDAGRAPH}")
    print()

    # PyTorch
    print("PyTorch Full Attention:")
    pytorch_results = benchmark_pytorch_attention(q, k, v, num_warmup, num_iters)
    print(f"  Total: {pytorch_results['total_ms']:.3f} ms")
    print()

    # FFA-Sample
    print("FFA-Sample (Triton) Sparse Attention:")
    k_sample = sample_k_fp16(k, BS=page_size)
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((1, num_blocks, num_kv_heads, head_dim), device=device, dtype=dtype)

    # Get kernel timings (without CUDAGraph for accurate individual kernel times)
    output, kernel_times = attn_forward_decode_sample4(
        q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
        k_full=k, v=v, BS=page_size, delta=delta,
        return_kernel_timings=True,
    )

    print(f"  Kernel breakdown (single call):")
    total_kernel_ms = 0
    for name, t in sorted(kernel_times.items()):
        if t is not None:
            print(f"    {name}: {t:.3f} ms")
            total_kernel_ms += t
    print(f"  Total kernel time: {total_kernel_ms:.3f} ms")

    # Full forward benchmark
    ffa_results = benchmark_ffa_sample(
        q, k, v, page_size=page_size, delta=delta,
        max_kept_ratio=max_kept_ratio, num_warmup=num_warmup, num_iters=num_iters,
        use_cudagraph=use_cudagraph,
    )

    print(f"\n  Forward time ({ffa_results.get('mode', 'unknown')}): {ffa_results['forward_ms']:.3f} ms")
    print(f"  Skip ratio: {ffa_results['skip_ratio']:.1%}")
    print(f"  Speedup vs PyTorch: {pytorch_results['total_ms'] / ffa_results['forward_ms']:.2f}x")
    print()


def run_delta_sensitivity(
    seq_len: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    deltas: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    max_kept_ratio: float = 0.5,
    num_warmup: int = 100,
    num_iters: int = 500,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """Test sensitivity to delta parameter."""

    print("=" * 80)
    print(f"Delta Sensitivity Analysis (seq_len={seq_len})")
    print("=" * 80)

    q = torch.randn(1, 1, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    k_sample = sample_k_fp16(k, BS=page_size)
    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((1, num_blocks, num_kv_heads, head_dim), device=device, dtype=dtype)

    # PyTorch baseline
    pytorch_results = benchmark_pytorch_attention(q, k, v, num_warmup, num_iters)
    pytorch_ms = pytorch_results["total_ms"]

    print(f"\nPyTorch baseline: {pytorch_ms:.3f} ms")
    print()

    print(f"{'Delta':>8} | {'Time(ms)':>10} | {'Skip Ratio':>10} | {'Kept Blocks':>12} | {'Speedup':>10}")
    print("-" * 65)

    for delta in deltas:
        # Get skip_ratio first
        _, skip_ratio = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=page_size, delta=delta,
            max_kept_ratio=max_kept_ratio, return_skip_ratio=True,
        )

        def run_fn():
            return attn_forward_decode_sample4(
                q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
                k_full=k, v=v, BS=page_size, delta=delta,
                max_kept_ratio=max_kept_ratio,
            )

        ffa_ms = benchmark_cuda_event(run_fn, iters=num_iters, warmup=num_warmup)

        kept_blocks = int(num_blocks * (1 - skip_ratio))
        speedup = pytorch_ms / ffa_ms

        print(f"{delta:>8.1f} | {ffa_ms:>10.3f} | {skip_ratio:>10.1%} | {kept_blocks:>12} | {speedup:>10.2f}x")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--max-kept-ratio", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--delta-sensitivity", action="store_true")
    parser.add_argument("--no-cudagraph", action="store_true", help="Disable CUDAGraph mode")

    args = parser.parse_args()

    use_cudagraph = not args.no_cudagraph

    print("\n" + "=" * 80)
    print("FFA-Sample Benchmark")
    print("Triton-based Sparse Attention with 4-point FP16 Sampling")
    print("=" * 80 + "\n")

    if args.detailed:
        run_detailed_benchmark(
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            delta=args.delta,
            max_kept_ratio=args.max_kept_ratio,
            num_warmup=args.warmup,
            num_iters=args.iters,
            use_cudagraph=use_cudagraph,
        )
    elif args.delta_sensitivity:
        run_delta_sensitivity(
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            max_kept_ratio=args.max_kept_ratio,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )
    else:
        run_benchmark(
            seq_lengths=args.seq_lens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            delta=args.delta,
            max_kept_ratio=args.max_kept_ratio,
            num_warmup=args.warmup,
            num_iters=args.iters,
            use_cudagraph=use_cudagraph,
        )

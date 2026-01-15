"""
Quest-only benchmark (在 quest conda 环境中运行)
"""

import sys
import time
import argparse

import torch

sys.path.insert(0, "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest")

import quest.utils


def benchmark_quest(
    seq_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    page_size: int = 16,
    page_budget_ratio: float = 0.2,
    num_warmup: int = 5,
    num_iters: int = 50,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """Benchmark Quest sparse attention."""
    num_layers = 1
    num_blocks = (seq_len + page_size - 1) // page_size
    page_budget = max(2, int(num_blocks * page_budget_ratio))
    max_seq_len = seq_len + 1024

    # Initialize controller
    controller = quest.utils.InferenceController(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    # Prefill
    k_prefill = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(seq_len)
    controller.begin_forward(seq_len)
    quest.utils.append_kv(k_prefill, v_prefill, controller, 0)
    controller.end_forward()

    # Decode
    q = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(1)
    controller.begin_forward(1)
    quest.utils.append_kv(k_decode, v_decode, controller, 0)

    # Warmup
    for _ in range(num_warmup):
        if controller.need_estimate():
            estimated = quest.utils.decode_estimate(q, controller, 0)
            quest.utils.decode_topk(estimated, controller)
            _ = quest.utils.decode_sparse_attn(q, controller, 0, controller.topk_dindices_buffer)
        else:
            _ = quest.utils.decode_sparse_attn(q, controller, 0, controller.kv_indices_without_last)
    torch.cuda.synchronize()

    # Benchmark
    results = {}

    if controller.need_estimate():
        # Estimate
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            estimated = quest.utils.decode_estimate(q, controller, 0)
        torch.cuda.synchronize()
        results["estimate_ms"] = (time.perf_counter() - start) / num_iters * 1000

        # TopK
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            quest.utils.decode_topk(estimated, controller)
        torch.cuda.synchronize()
        results["topk_ms"] = (time.perf_counter() - start) / num_iters * 1000

        # Sparse attention
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = quest.utils.decode_sparse_attn(q, controller, 0, controller.topk_dindices_buffer)
        torch.cuda.synchronize()
        results["sparse_attn_ms"] = (time.perf_counter() - start) / num_iters * 1000

        results["total_ms"] = results["estimate_ms"] + results["topk_ms"] + results["sparse_attn_ms"]
    else:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = quest.utils.decode_sparse_attn(q, controller, 0, controller.kv_indices_without_last)
        torch.cuda.synchronize()
        results["sparse_attn_ms"] = (time.perf_counter() - start) / num_iters * 1000
        results["total_ms"] = results["sparse_attn_ms"]

    controller.end_forward()
    controller.clean_states()

    return results, page_budget, num_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-budget-ratio", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print("=" * 80)
    print("Quest Sparse Attention Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_heads: {args.num_heads}, head_dim: {args.head_dim}")
    print(f"  page_size: {args.page_size}, page_budget_ratio: {args.page_budget_ratio}")
    print(f"  warmup: {args.warmup}, iterations: {args.iters}")
    print()

    print(f"{'SeqLen':>8} | {'Blocks':>7} | {'Budget':>7} | {'Estimate':>10} | {'TopK':>10} | {'Attn':>10} | {'Total':>10}")
    print("-" * 85)

    for seq_len in args.seq_lens:
        try:
            results, page_budget, num_blocks = benchmark_quest(
                seq_len=seq_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                page_size=args.page_size,
                page_budget_ratio=args.page_budget_ratio,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )

            estimate_ms = results.get("estimate_ms", 0)
            topk_ms = results.get("topk_ms", 0)
            attn_ms = results.get("sparse_attn_ms", 0)
            total_ms = results.get("total_ms", 0)

            print(f"{seq_len:>8} | {num_blocks:>7} | {page_budget:>7} | {estimate_ms:>10.3f} | {topk_ms:>10.3f} | {attn_ms:>10.3f} | {total_ms:>10.3f}")

        except Exception as e:
            print(f"{seq_len:>8} | Failed: {e}")

        torch.cuda.empty_cache()

    print()


if __name__ == "__main__":
    main()

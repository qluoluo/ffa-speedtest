"""
速度对比测试: Quest vs FFA-Sample (Triton Sample4 FP16)

对比两种稀疏注意力方法的性能:
- Quest: 使用 min/max K 估计 + CUDA kernel
- FFA-Sample: 使用 4 点 FP16 采样 + Triton kernel
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# 添加路径 - Quest 需要先添加到 sys.path
QUEST_PATH = "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest"
FFA_SAMPLE_PATH = "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-flashinfer-sample"

# 确保路径在最前面
if QUEST_PATH not in sys.path:
    sys.path.insert(0, QUEST_PATH)
if FFA_SAMPLE_PATH not in sys.path:
    sys.path.insert(0, FFA_SAMPLE_PATH)

# 设置 LD_LIBRARY_PATH (如果需要)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def benchmark_quest(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_layers: int,
    page_size: int,
    page_budget: int,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> Dict[str, float]:
    """
    Benchmark Quest sparse attention.

    Quest 需要完整的 InferenceController 流程:
    1. prepare_metadata
    2. begin_forward
    3. append_kv (prefill)
    4. decode: estimate -> topk -> sparse_attn
    """
    import quest.utils

    B, T, num_heads, head_dim = k.shape
    device = k.device
    dtype = k.dtype
    max_seq_len = T + 1024  # 留一些余量

    results = {}

    # 初始化 Controller
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

    # Prefill 阶段: 将 K, V 写入 cache
    k_prefill = k.squeeze(0)  # [T, H, D]
    v_prefill = v.squeeze(0)

    controller.prepare_metadata(T)
    controller.begin_forward(T)
    quest.utils.append_kv(k_prefill, v_prefill, controller, 0)
    controller.end_forward()

    # Decode 阶段
    q_decode = q.squeeze(0).squeeze(0)  # [H, D]
    q_decode = q_decode.unsqueeze(0)  # [1, H, D]

    # 添加一个 decode token
    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(1)
    controller.begin_forward(1)
    quest.utils.append_kv(k_decode, v_decode, controller, 0)

    # Warmup
    for _ in range(num_warmup):
        if controller.need_estimate():
            estimated = quest.utils.decode_estimate(q_decode, controller, 0)
            quest.utils.decode_topk(estimated, controller)
            _ = quest.utils.decode_sparse_attn(
                q_decode, controller, 0,
                controller.topk_dindices_buffer
            )
        else:
            _ = quest.utils.decode_sparse_attn(
                q_decode, controller, 0,
                controller.kv_indices_without_last
            )
    torch.cuda.synchronize()

    # Benchmark estimate
    if controller.need_estimate():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            estimated = quest.utils.decode_estimate(q_decode, controller, 0)
        torch.cuda.synchronize()
        results["estimate_ms"] = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark topk
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            quest.utils.decode_topk(estimated, controller)
        torch.cuda.synchronize()
        results["topk_ms"] = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark sparse_attn
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = quest.utils.decode_sparse_attn(
                q_decode, controller, 0,
                controller.topk_dindices_buffer
            )
        torch.cuda.synchronize()
        results["sparse_attn_ms"] = (time.perf_counter() - start) / num_iters * 1000

        results["total_ms"] = results["estimate_ms"] + results["topk_ms"] + results["sparse_attn_ms"]
    else:
        # 不需要估计，直接全注意力
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = quest.utils.decode_sparse_attn(
                q_decode, controller, 0,
                controller.kv_indices_without_last
            )
        torch.cuda.synchronize()
        results["sparse_attn_ms"] = (time.perf_counter() - start) / num_iters * 1000
        results["total_ms"] = results["sparse_attn_ms"]

    controller.end_forward()
    controller.clean_states()

    return results


def benchmark_ffa_sample(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    delta: float = 5.0,
    max_kept_ratio: float = 0.2,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> Dict[str, float]:
    """
    Benchmark FFA-Sample (Triton Sample4 FP16) sparse attention.
    """
    from ffa_sample.kernels import sample_k_fp16, attn_forward_decode_sample4

    B, T, HKV, K = k.shape
    device = k.device
    dtype = k.dtype

    results = {}

    # 预处理: 提取采样 K
    torch.cuda.synchronize()
    start = time.perf_counter()
    k_sample = sample_k_fp16(k, BS=page_size)
    torch.cuda.synchronize()
    results["sample_k_ms"] = (time.perf_counter() - start) * 1000

    num_blocks = k_sample.shape[1]
    k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=page_size, delta=delta,
            max_kept_ratio=max_kept_ratio,
        )
    torch.cuda.synchronize()

    # Benchmark full forward (including threshold + stage1 + stage2)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        output, skip_ratio = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=page_size, delta=delta,
            max_kept_ratio=max_kept_ratio, return_skip_ratio=True,
        )
    torch.cuda.synchronize()
    results["forward_ms"] = (time.perf_counter() - start) / num_iters * 1000
    results["skip_ratio"] = skip_ratio

    # Benchmark with kernel timings
    output, kernel_times = attn_forward_decode_sample4(
        q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
        k_full=k, v=v, BS=page_size, delta=delta,
        max_kept_ratio=max_kept_ratio, return_kernel_timings=True,
    )

    for name, t in kernel_times.items():
        if t is not None:
            results[f"kernel_{name}_ms"] = t

    results["total_ms"] = results["forward_ms"]

    return results


def benchmark_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> Dict[str, float]:
    """
    Benchmark standard PyTorch attention as baseline.
    """
    B, Tq, HQ, K = q.shape
    _, T, HKV, _ = k.shape
    V = v.shape[-1]

    G = HQ // HKV
    scale = 1.0 / (K ** 0.5)

    # 扩展 K, V 以匹配 Q 头数 (GQA)
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, K)
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, G, -1).reshape(B, T, HQ, V)

    q_2d = q.squeeze(1)  # [B, HQ, K]

    # Warmup
    for _ in range(num_warmup):
        scores = torch.einsum("bhk,bthk->bht", q_2d, k_exp) * scale
        attn_weights = F.softmax(scores, dim=-1)
        _ = torch.einsum("bht,bthv->bhv", attn_weights, v_exp)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        scores = torch.einsum("bhk,bthk->bht", q_2d, k_exp) * scale
        attn_weights = F.softmax(scores, dim=-1)
        _ = torch.einsum("bht,bthv->bhv", attn_weights, v_exp)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000

    return {"total_ms": elapsed}


def run_benchmark(
    seq_lengths: List[int],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    page_budget_ratio: float = 0.2,
    delta: float = 5.0,
    num_warmup: int = 5,
    num_iters: int = 50,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """运行完整的 benchmark."""

    print("=" * 80)
    print("Sparse Attention Benchmark: Quest vs FFA-Sample vs PyTorch")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  page_size: {page_size}, page_budget_ratio: {page_budget_ratio}")
    print(f"  delta: {delta}, dtype: {dtype}")
    print(f"  warmup: {num_warmup}, iterations: {num_iters}")
    print()

    # 表头
    print(f"{'SeqLen':>8} | {'PyTorch':>10} | {'Quest':>10} | {'FFA-Sample':>10} | {'Quest Skip':>10} | {'FFA Skip':>10} | {'Quest Speedup':>12} | {'FFA Speedup':>12}")
    print("-" * 110)

    for seq_len in seq_lengths:
        num_blocks = (seq_len + page_size - 1) // page_size
        page_budget = max(2, int(num_blocks * page_budget_ratio))

        # 创建输入
        q = torch.randn(1, 1, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

        # PyTorch baseline
        try:
            pytorch_results = benchmark_pytorch_attention(
                q, k, v, num_warmup=num_warmup, num_iters=num_iters
            )
            pytorch_ms = pytorch_results["total_ms"]
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            pytorch_ms = float('nan')

        # Quest
        try:
            quest_results = benchmark_quest(
                q, k, v,
                num_layers=1,
                page_size=page_size,
                page_budget=page_budget,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            quest_ms = quest_results["total_ms"]
            quest_skip = "N/A"  # Quest 使用 page_budget 控制
        except Exception as e:
            print(f"Quest benchmark failed for seq_len={seq_len}: {e}")
            quest_ms = float('nan')
            quest_skip = "N/A"

        # FFA-Sample
        try:
            ffa_results = benchmark_ffa_sample(
                q, k, v,
                page_size=page_size,
                delta=delta,
                max_kept_ratio=page_budget_ratio,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            ffa_ms = ffa_results["total_ms"]
            ffa_skip = f"{ffa_results.get('skip_ratio', 0):.1%}"
        except Exception as e:
            print(f"FFA-Sample benchmark failed for seq_len={seq_len}: {e}")
            ffa_ms = float('nan')
            ffa_skip = "N/A"

        # 计算加速比
        quest_speedup = pytorch_ms / quest_ms if quest_ms > 0 else float('nan')
        ffa_speedup = pytorch_ms / ffa_ms if ffa_ms > 0 else float('nan')

        print(f"{seq_len:>8} | {pytorch_ms:>10.3f} | {quest_ms:>10.3f} | {ffa_ms:>10.3f} | {quest_skip:>10} | {ffa_skip:>10} | {quest_speedup:>12.2f}x | {ffa_speedup:>12.2f}x")

        # 清理显存
        del q, k, v
        torch.cuda.empty_cache()

    print()


def run_detailed_benchmark(
    seq_len: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    page_budget_ratio: float = 0.2,
    delta: float = 5.0,
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """运行详细的单配置 benchmark，展示各阶段耗时."""

    print("=" * 80)
    print(f"Detailed Benchmark (seq_len={seq_len})")
    print("=" * 80)

    num_blocks = (seq_len + page_size - 1) // page_size
    page_budget = max(2, int(num_blocks * page_budget_ratio))

    q = torch.randn(1, 1, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    print(f"\nConfiguration:")
    print(f"  seq_len: {seq_len}, num_blocks: {num_blocks}, page_budget: {page_budget}")
    print()

    # PyTorch
    print("PyTorch Attention:")
    pytorch_results = benchmark_pytorch_attention(q, k, v, num_warmup, num_iters)
    print(f"  Total: {pytorch_results['total_ms']:.3f} ms")
    print()

    # Quest
    print("Quest Sparse Attention:")
    try:
        quest_results = benchmark_quest(
            q, k, v, num_layers=1, page_size=page_size,
            page_budget=page_budget, num_warmup=num_warmup, num_iters=num_iters
        )
        for key, val in sorted(quest_results.items()):
            print(f"  {key}: {val:.3f} ms" if isinstance(val, float) else f"  {key}: {val}")
    except Exception as e:
        print(f"  Failed: {e}")
    print()

    # FFA-Sample
    print("FFA-Sample (Triton) Sparse Attention:")
    try:
        ffa_results = benchmark_ffa_sample(
            q, k, v, page_size=page_size, delta=delta,
            max_kept_ratio=page_budget_ratio, num_warmup=num_warmup, num_iters=num_iters
        )
        for key, val in sorted(ffa_results.items()):
            if isinstance(val, float):
                print(f"  {key}: {val:.3f} ms" if 'ms' in key else f"  {key}: {val:.2%}")
            else:
                print(f"  {key}: {val}")
    except Exception as e:
        print(f"  Failed: {e}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sparse attention methods")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384],
                        help="Sequence lengths to test")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)  # Quest 不支持 GQA，需要 num_kv_heads == num_heads
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--page-budget-ratio", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--detailed", action="store_true", help="Run detailed single benchmark")
    parser.add_argument("--detailed-seq-len", type=int, default=4096)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Sparse Attention Benchmark")
    print("Quest (CUDA + FlashInfer) vs FFA-Sample (Triton)")
    print("=" * 80 + "\n")

    if args.detailed:
        run_detailed_benchmark(
            seq_len=args.detailed_seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            page_budget_ratio=args.page_budget_ratio,
            delta=args.delta,
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
            page_budget_ratio=args.page_budget_ratio,
            delta=args.delta,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )

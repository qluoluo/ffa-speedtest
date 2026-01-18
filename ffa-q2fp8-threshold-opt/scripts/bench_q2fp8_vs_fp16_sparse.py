#!/usr/bin/env python3
"""
Benchmark: Compare Q2FP8 (int2+fp8) vs FP16 attention at the same sparsity levels.

This script measures how much slower int2+fp8 KV attention is compared to fp16 KV attention
under the same sparsity conditions (i.e., both methods skip the same tokens).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils.bench import benchmark


# ============================================================================
# FP16 Sparse Attention Kernel (baseline)
# ============================================================================
@triton.jit
def attn_forward_fp16_sparse_stage1(
    q, k, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    """Stage1: Compute attention scores and output for each block, respecting mask."""
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k = tl.arange(0, K)

    # Load q [BM_DOT, K]
    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        tb_sb = pid_tb * NSB + sb

        # Check if this block is marked for processing
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb).to(tl.int1)
        if keep:
            # Load k [K, SBS]
            k_ptrs = k + pid_b * (T * HKV * K) + offs_t_sb[None, :] * (HKV * K) + pid_hkv * K + offs_k[:, None]
            k_tile = tl.load(k_ptrs, mask=t_mask_sb[None, :], other=0.0).to(tl.float16)

            # Compute attention scores
            b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
            b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
            m_rows = tl.max(b_s, axis=1)

            b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            # Load v and compute output
            v_offs = tl.arange(0, V)
            v_ptrs = v + pid_b * (T * HKV * V) + offs_t_sb[:, None] * (HKV * V) + pid_hkv * V + v_offs[None, :]
            v_tile = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), v_tile, out_dtype=tl.float32)

            # Store intermediate results
            m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])


@triton.jit
def attn_forward_fp16_sparse_stage2(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    """Stage2: Reduce across all blocks to get final output."""
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def attn_forward_fp16_sparse(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k: torch.Tensor,           # [B, T, HKV, K]
    v: torch.Tensor,           # [B, T, HKV, V]
    mask_buf: torch.Tensor,    # [B, HKV, NTBS] - pre-computed mask from Q2FP8
    BS: int = 128,
    SBS: int | None = None,
    scale: float = None,
):
    """FP16 sparse attention using the same mask pattern as Q2FP8."""
    B, Tq, HQ, K = q.shape
    _, T, HKV, _ = k.shape
    _, _, _, V = v.shape
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    q = q.squeeze(1).contiguous()  # [B, HQ, K]
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)

    attn_forward_fp16_sparse_stage1[(NTB, B, HKV)](
        q, k, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS,
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    attn_forward_fp16_sparse_stage2[(B, HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
    )

    return o


# ============================================================================
# Helper functions
# ============================================================================
def generate_random_mask(B, HKV, NTBS, sparsity: float, device):
    """Generate a random mask with specified sparsity (0.0 = no skip, 1.0 = skip all)."""
    # sparsity = fraction of blocks to SKIP
    # So keep_prob = 1.0 - sparsity
    keep_prob = 1.0 - sparsity
    mask = (torch.rand(B, HKV, NTBS, device=device) < keep_prob).to(torch.int8)
    return mask


def quantize_k_2bit_fp8_residual_symmetric(
    k: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e5m2,
    k_bits: int = 2,
):
    """Symmetric 2-bit quantization with FP8 residual."""
    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    k_absmax = k.abs().amax(dim=1)
    scale = (k_absmax / qzero).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)
    k_dequant = (k_q.to(torch.float32) - qzero) * scale[:, None, :, :].to(torch.float32)
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    B, T, HKV, K = k_q.shape
    values_per_byte = 4
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)
        k_q = torch.cat([k_q, pad_tensor], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()
    return k_q_packed, scale, k_residual


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Q2FP8 (int2+fp8) vs FP16 attention at the same sparsity levels."
    )
    p.add_argument("--B", type=int, default=1, help="Batch size")
    p.add_argument("--T", type=int, default=65536, help="Sequence length")
    p.add_argument("--HQ", type=int, default=32, help="Number of query heads")
    p.add_argument("--HKV", type=int, default=8, help="Number of KV heads")
    p.add_argument("--K", type=int, default=128, help="Head dimension for K")
    p.add_argument("--V", type=int, default=128, help="Head dimension for V")
    p.add_argument("--BS", type=int, default=128, help="Block size")
    p.add_argument("--SBS", type=int, default=None, help="Sub-block size")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument(
        "--sparsity",
        type=str,
        default="0.0,0.25,0.5,0.75,0.9,0.95,0.99",
        help="Comma-separated sparsity levels to test (0.0=dense, 1.0=fully sparse)"
    )
    p.add_argument("--iters", type=int, default=200, help="Benchmark iterations per round")
    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    p.add_argument("--rounds", type=int, default=5, help="Number of rounds to run, take median")
    p.add_argument("--delta", type=float, default=5.0, help="Delta for Q2FP8 threshold")
    p.add_argument("--output", type=str, default=None, help="Output JSON file path")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    # Import Q2FP8 kernel with external mask (fair comparison - both use same mask)
    from attn_kernel.attn_q2fp8_ext_mask import attn_forward_q2fp8_with_ext_mask

    B = args.B
    T = args.T
    HQ = args.HQ
    HKV = args.HKV
    K = args.K
    V = args.V
    BS = args.BS
    SBS = args.SBS if args.SBS else BS
    dtype = map_dtype(args.dtype)
    delta = args.delta

    sparsity_levels = [float(s.strip()) for s in args.sparsity.split(",")]

    device = torch.device("cuda")
    scale = 1.0 / math.sqrt(K)

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    print(f"[Config] B={B}, T={T}, HQ={HQ}, HKV={HKV}, K={K}, V={V}")
    print(f"[Config] BS={BS}, SBS={SBS}, NTB={NTB}, NTBS={NTBS}")
    print(f"[Config] dtype={args.dtype}, iters={args.iters}, warmup={args.warmup}, rounds={args.rounds}")
    print(f"[Config] Sparsity levels: {sparsity_levels}")
    print()

    # Generate random inputs
    q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

    # Quantize K for Q2FP8
    k_q, k_scale, k_residual = quantize_k_2bit_fp8_residual_symmetric(k)

    rounds = args.rounds

    def benchmark_median(fn):
        """Run benchmark multiple rounds and return median."""
        times = []
        for _ in range(rounds):
            t = benchmark(fn, iters=args.iters, warmup=args.warmup)
            times.append(t)
        times.sort()
        return times[rounds // 2]  # median

    results = []

    for sparsity in tqdm(sparsity_levels, desc="Testing sparsity levels"):
        # Generate mask with specified sparsity
        # mask_buf[b, hkv, tb] = 1 means KEEP this block, 0 means SKIP
        mask_buf = generate_random_mask(B, HKV, NTBS, sparsity, device)
        actual_keep_ratio = mask_buf.float().mean().item()
        actual_sparsity = 1.0 - actual_keep_ratio

        # Benchmark FP16 sparse attention
        def run_fp16():
            return attn_forward_fp16_sparse(
                q, k, v,
                mask_buf,
                BS=BS,
                SBS=SBS,
                scale=scale,
            )

        # Benchmark Q2FP8 attention with external mask (same mask as FP16)
        def run_q2fp8():
            return attn_forward_q2fp8_with_ext_mask(
                q=q,
                k_q=k_q,
                k_scale=k_scale,
                v=v,
                ext_mask=mask_buf,
                k_residual=k_residual,
                k_bits=2,
                scale=scale,
                BS=BS,
                SBS=SBS,
                use_fp8_residual=True,
            )

        # Warmup and benchmark (multiple rounds, take median)
        ms_fp16 = benchmark_median(run_fp16)
        ms_q2fp8 = benchmark_median(run_q2fp8)

        slowdown = ms_q2fp8 / ms_fp16 if ms_fp16 > 0 else float("inf")

        result = {
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "fp16_ms": ms_fp16,
            "q2fp8_ms": ms_q2fp8,
            "slowdown": slowdown,
        }
        results.append(result)

        print(
            f"Sparsity={sparsity:.2f} (actual={actual_sparsity:.3f}): "
            f"FP16={ms_fp16:.4f}ms, Q2FP8={ms_q2fp8:.4f}ms, "
            f"Slowdown={slowdown:.2f}x"
        )

    # Also run dense baseline (no sparsity)
    print("\n[Dense Baseline]")
    mask_dense = torch.ones(B, HKV, NTBS, device=device, dtype=torch.int8)

    def run_fp16_dense():
        return attn_forward_fp16_sparse(q, k, v, mask_dense, BS=BS, SBS=SBS, scale=scale)

    def run_q2fp8_dense():
        return attn_forward_q2fp8_with_ext_mask(
            q=q, k_q=k_q, k_scale=k_scale, v=v, ext_mask=mask_dense,
            k_residual=k_residual, k_bits=2, scale=scale, BS=BS, SBS=SBS,
            use_fp8_residual=True,
        )

    ms_fp16_dense = benchmark_median(run_fp16_dense)
    ms_q2fp8_dense = benchmark_median(run_q2fp8_dense)
    slowdown_dense = ms_q2fp8_dense / ms_fp16_dense if ms_fp16_dense > 0 else float("inf")

    print(f"Dense: FP16={ms_fp16_dense:.4f}ms, Q2FP8={ms_q2fp8_dense:.4f}ms, Slowdown={slowdown_dense:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Q2FP8 slowdown compared to FP16 at same sparsity")
    print("=" * 60)
    print(f"{'Sparsity':>10} | {'FP16 (ms)':>12} | {'Q2FP8 (ms)':>12} | {'Slowdown':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['target_sparsity']:>10.2f} | {r['fp16_ms']:>12.4f} | {r['q2fp8_ms']:>12.4f} | {r['slowdown']:>10.2f}x"
        )
    print("-" * 60)
    print(f"{'Dense':>10} | {ms_fp16_dense:>12.4f} | {ms_q2fp8_dense:>12.4f} | {slowdown_dense:>10.2f}x")

    # Save results
    output_data = {
        "config": {
            "B": B,
            "T": T,
            "HQ": HQ,
            "HKV": HKV,
            "K": K,
            "V": V,
            "BS": BS,
            "SBS": SBS,
            "dtype": args.dtype,
            "iters": args.iters,
            "warmup": args.warmup,
            "delta": delta,
        },
        "results": results,
        "dense_baseline": {
            "fp16_ms": ms_fp16_dense,
            "q2fp8_ms": ms_q2fp8_dense,
            "slowdown": slowdown_dense,
        },
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = THIS_DIR / "bench_q2fp8_vs_fp16_sparse_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Info] Results saved to {output_path}")

    # Plot if requested
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            sparsities = [r["target_sparsity"] for r in results]
            fp16_times = [r["fp16_ms"] for r in results]
            q2fp8_times = [r["q2fp8_ms"] for r in results]
            slowdowns = [r["slowdown"] for r in results]

            # Plot latency comparison
            ax1.plot(sparsities, fp16_times, "o-", label="FP16", color="blue", markersize=6)
            ax1.plot(sparsities, q2fp8_times, "s-", label="Q2FP8 (int2+fp8)", color="red", markersize=6)
            ax1.axhline(y=ms_fp16_dense, color="blue", linestyle="--", alpha=0.5, label="FP16 Dense")
            ax1.axhline(y=ms_q2fp8_dense, color="red", linestyle="--", alpha=0.5, label="Q2FP8 Dense")
            ax1.set_xlabel("Sparsity (fraction of blocks skipped)")
            ax1.set_ylabel("Latency (ms)")
            ax1.set_title(f"Latency vs Sparsity (T={T}, HQ={HQ}, HKV={HKV})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot slowdown
            ax2.plot(sparsities, slowdowns, "o-", color="green", markersize=6)
            ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No slowdown")
            ax2.axhline(y=slowdown_dense, color="orange", linestyle="--", alpha=0.5, label=f"Dense slowdown ({slowdown_dense:.2f}x)")
            ax2.set_xlabel("Sparsity (fraction of blocks skipped)")
            ax2.set_ylabel("Slowdown (Q2FP8 / FP16)")
            ax2.set_title("Q2FP8 Slowdown vs FP16 at Same Sparsity")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            plot_path = output_path.with_suffix(".png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[Info] Plot saved to {plot_path}")

        except Exception as e:
            print(f"[Warning] Could not create plot: {e}")


if __name__ == "__main__":
    main()

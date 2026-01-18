# Benchmark: Per-Block Scale vs Global Scale for Q2FP8 Attention
import argparse
import math
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.bench import benchmark
from utils.load import load_qkvh

EXP_ROOT_DIR = ROOT_DIR / "attn_analysis" / "result"
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Per-Block vs Global Scale Q2FP8")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--BS", type=int, default=128, help="Block size for attention")
    p.add_argument("--SBS", type=int, default=None, help="Sub-block size (default=BS)")
    p.add_argument("--delta", type=float, default=5.0, help="Threshold delta")
    p.add_argument("--layer", type=int, default=1, help="Layer index")
    p.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    p.add_argument("--step", type=int, default=8192, help="Length step")
    p.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    p.add_argument("--force", action="store_true", help="Force rerun")
    return p.parse_args()


def map_dtype(dtype_str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


def convert_layout(q_rope_1, k_rope, v):
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    q = q_rope_1[:, :, 0, :].contiguous()
    k = k_rope.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def quantize_global_scale(k, k_bits=2, fp8_dtype=torch.float8_e5m2):
    """Global scale: k_scale shape [B, HKV, K]"""
    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    # Global: max over all tokens
    k_absmax = k.abs().amax(dim=1)  # [B, HKV, K]
    scale = (k_absmax / qzero).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)
    k_dequant = (k_q.to(torch.float32) - qzero) * scale[:, None, :, :].to(torch.float32)
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack 4x2-bit into 1 byte
    B, T, HKV, K = k_q.shape
    values_per_byte = 4
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        k_q = torch.cat([k_q, torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (k_q[..., 0] | (k_q[..., 1] << 2) | (k_q[..., 2] << 4) | (k_q[..., 3] << 6)).contiguous()
    return k_q_packed, scale, k_residual


def quantize_perblock_scale(k, BS, k_bits=2, fp8_dtype=torch.float8_e5m2):
    """Per-block scale: k_scale shape [B, NTB, HKV, K]"""
    B, T, HKV, K = k.shape
    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    NTB = (T + BS - 1) // BS

    # Pad to multiple of BS
    pad_T = NTB * BS - T
    if pad_T > 0:
        k_padded = torch.cat([k, torch.zeros((B, pad_T, HKV, K), device=k.device, dtype=k.dtype)], dim=1)
    else:
        k_padded = k

    # Reshape to blocks: [B, NTB, BS, HKV, K]
    k_blocks = k_padded.view(B, NTB, BS, HKV, K)

    # Per-block absmax: [B, NTB, HKV, K]
    k_absmax = k_blocks.abs().amax(dim=2)
    scale = (k_absmax / qzero).clamp_min(1e-6).contiguous()

    # Quantize with per-block scale
    k_q_blocks = torch.round(k_blocks / scale[:, :, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)
    k_dequant_blocks = (k_q_blocks.to(torch.float32) - qzero) * scale[:, :, None, :, :].to(torch.float32)
    k_residual_blocks = (k_blocks.to(torch.float32) - k_dequant_blocks).to(fp8_dtype)

    # Reshape back: [B, T_padded, HKV, K]
    k_q = k_q_blocks.view(B, NTB * BS, HKV, K)[:, :T, :, :].contiguous()
    k_residual = k_residual_blocks.view(B, NTB * BS, HKV, K)[:, :T, :, :].contiguous()

    # Pack 4x2-bit into 1 byte
    values_per_byte = 4
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        k_q = torch.cat([k_q, torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (k_q[..., 0] | (k_q[..., 1] << 2) | (k_q[..., 2] << 4) | (k_q[..., 3] << 6)).contiguous()

    return k_q_packed, scale, k_residual


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    # Import kernel modules
    from attn_kernel.attn_q2fp8_sym_mask import (
        attn_forward_decode_quantized as attn_sym_mask,
    )
    from attn_kernel.attn_q2fp8_asym_mask import (
        attn_forward_decode_quantized as attn_asym_mask,
    )

    dtype = map_dtype(args.dtype)
    BS = args.BS
    SBS = args.SBS if args.SBS else BS
    delta = args.delta

    # Load data
    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"

    print(f"[Info] Loading layer {args.layer} data...")
    data_iter = load_qkvh(layer_data_root, device="cuda", start_layer=args.layer, max_length=args.max_length)
    layer_data = next(data_iter)

    q_rope_full = layer_data["q_rope"].to(dtype=dtype)
    k_rope_full = layer_data["k_rope"].to(dtype=dtype)
    v_full = layer_data["v"].to(dtype=dtype)

    B, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] B={B}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")
    print(f"[Info] BS={BS}, SBS={SBS}, delta={delta}")

    lengths = list(range(args.step, T_full, args.step)) + [T_full]

    results = {
        "lengths": [],
        "sym_global_ms": [],
        "sym_perblock_ms": [],
        "asym_global_ms": [],
        "asym_perblock_ms": [],
    }

    for L in tqdm(lengths, desc="Benchmarking"):
        q_rope_1 = q_rope_full[:, :, L-1:L, :].contiguous()
        k_rope = k_rope_full[:, :, :L, :].contiguous()
        v = v_full[:, :, :L, :].contiguous()

        q, k, v_conv = convert_layout(q_rope_1, k_rope, v)
        q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]

        # Global scale quantization
        k_q_global, k_scale_global, k_res_global = quantize_global_scale(k)

        # Per-block scale quantization
        k_q_perblock, k_scale_perblock, k_res_perblock = quantize_perblock_scale(k, BS)

        NTB = (L + BS - 1) // BS

        # Test sym_mask with global scale
        def run_sym_global():
            return attn_sym_mask(
                q=q_1, k_q=k_q_global, k_scale=k_scale_global, k_residual=k_res_global, v=v_conv,
                k_bits=2, scale=scale, BS=BS, SBS=SBS, delta=delta,
            )

        # Test sym_mask with per-block scale
        def run_sym_perblock():
            return attn_sym_mask(
                q=q_1, k_q=k_q_perblock, k_scale=k_scale_perblock, k_residual=k_res_perblock, v=v_conv,
                k_bits=2, scale=scale, BS=BS, SBS=SBS, delta=delta,
            )

        # For asym_mask, we need k_min as well
        # Global asymmetric quantization
        k_min_global = k.amin(dim=1)  # [B, HKV, K]
        k_max_global = k.amax(dim=1)
        k_scale_asym_global = ((k_max_global - k_min_global).clamp_min(1e-6) / 3.0).contiguous()

        # Per-block asymmetric quantization
        k_padded = k
        pad_T = NTB * BS - L
        if pad_T > 0:
            k_padded = torch.cat([k, torch.zeros((B, pad_T, Hkv, K), device=k.device, dtype=k.dtype)], dim=1)
        k_blocks = k_padded.view(B, NTB, BS, Hkv, K)
        k_min_perblock = k_blocks.amin(dim=2)  # [B, NTB, HKV, K]
        k_max_perblock = k_blocks.amax(dim=2)
        k_scale_asym_perblock = ((k_max_perblock - k_min_perblock).clamp_min(1e-6) / 3.0).contiguous()

        # Test asym_mask with global scale
        def run_asym_global():
            return attn_asym_mask(
                q=q_1, k_q=k_q_global, k_scale=k_scale_asym_global, k_min=k_min_global,
                k_residual=k_res_global, v=v_conv,
                k_bits=2, scale=scale, BS=BS, SBS=SBS, delta=delta,
            )

        # Test asym_mask with per-block scale
        def run_asym_perblock():
            return attn_asym_mask(
                q=q_1, k_q=k_q_perblock, k_scale=k_scale_asym_perblock, k_min=k_min_perblock,
                k_residual=k_res_perblock, v=v_conv,
                k_bits=2, scale=scale, BS=BS, SBS=SBS, delta=delta,
            )

        # Warmup
        for _ in range(3):
            run_sym_global()
            run_sym_perblock()
            run_asym_global()
            run_asym_perblock()
        torch.cuda.synchronize()

        # Benchmark
        ms_sym_global = benchmark(run_sym_global, iters=args.iters, warmup=args.warmup)
        ms_sym_perblock = benchmark(run_sym_perblock, iters=args.iters, warmup=args.warmup)
        ms_asym_global = benchmark(run_asym_global, iters=args.iters, warmup=args.warmup)
        ms_asym_perblock = benchmark(run_asym_perblock, iters=args.iters, warmup=args.warmup)

        results["lengths"].append(L)
        results["sym_global_ms"].append(ms_sym_global)
        results["sym_perblock_ms"].append(ms_sym_perblock)
        results["asym_global_ms"].append(ms_asym_global)
        results["asym_perblock_ms"].append(ms_asym_perblock)

        print(f"  L={L}: sym_global={ms_sym_global:.3f}ms, sym_perblock={ms_sym_perblock:.3f}ms, "
              f"asym_global={ms_asym_global:.3f}ms, asym_perblock={ms_asym_perblock:.3f}ms")

    # Save results
    output_dir = ROOT_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"perblock_vs_global_BS{BS}_delta{delta}_layer{args.layer}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Info] Saved results to {result_file}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Symmetric quantization comparison
        ax1.plot(results["lengths"], results["sym_global_ms"], 'b-o', label="Sym Global Scale", markersize=4)
        ax1.plot(results["lengths"], results["sym_perblock_ms"], 'r-s', label="Sym Per-Block Scale", markersize=4)
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title(f"Symmetric Quantization: Global vs Per-Block Scale\n(BS={BS}, delta={delta})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Asymmetric quantization comparison
        ax2.plot(results["lengths"], results["asym_global_ms"], 'b-o', label="Asym Global Scale", markersize=4)
        ax2.plot(results["lengths"], results["asym_perblock_ms"], 'r-s', label="Asym Per-Block Scale", markersize=4)
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title(f"Asymmetric Quantization: Global vs Per-Block Scale\n(BS={BS}, delta={delta})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / f"perblock_vs_global_BS{BS}_delta{delta}_layer{args.layer}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"[Info] Saved plot to {plot_file}")

    except Exception as e:
        print(f"[Warning] Failed to plot: {e}")

    # Print summary
    print("\n=== Summary ===")
    final_idx = -1
    print(f"At T={results['lengths'][final_idx]}:")
    print(f"  Symmetric:  Global={results['sym_global_ms'][final_idx]:.3f}ms, "
          f"PerBlock={results['sym_perblock_ms'][final_idx]:.3f}ms, "
          f"Diff={results['sym_perblock_ms'][final_idx] - results['sym_global_ms'][final_idx]:.3f}ms")
    print(f"  Asymmetric: Global={results['asym_global_ms'][final_idx]:.3f}ms, "
          f"PerBlock={results['asym_perblock_ms'][final_idx]:.3f}ms, "
          f"Diff={results['asym_perblock_ms'][final_idx] - results['asym_global_ms'][final_idx]:.3f}ms")


if __name__ == "__main__":
    main()

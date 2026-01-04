#!/usr/bin/env python3
"""
H100优化测试脚本 - 使用已dump的真实LLM数据

这个脚本从已经dump的真实LLM数据中加载Q, K, V，然后进行benchmark测试。
可以获得真实的skip ratio（预期99%+）和准确的性能数据。
"""

import argparse
import json
import math
import sys
from pathlib import Path
import torch
from typing import Dict, List, Tuple

# Add project root to path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized
from utils.flash import flash_attn_compute
from utils.load import load_qkvh


def quantize_k_2bit_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2):
    """Quantize K to 2-bit with FP8 residual."""
    # Scale/zero are per (B, HKV, K); token dimension is removed
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack to 2-bit
    values_per_byte = 4
    B, T, HKV, K = k_q.shape
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
    return k_q_packed, scale, zero, k_residual


def convert_layout(q_rope: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor):
    """
    Convert from saved layout to kernel expected layout.

    Saved: q_rope [B, HQ, T, K], k_rope [B, HKV, T, K], v [B, HKV, T, V]
    Kernel expects: q [B, 1, HQ, K], k [B, T, HKV, K], v [B, T, HKV, V]
    """
    B, Hq, T, K = q_rope.shape
    Bk, Hkv, Tk, Kk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape

    assert B == Bk == Bv and T == Tk == Tv and Hkv == Hvv

    # Take last token for q (decode scenario)
    q = q_rope[:, :, -1:, :].transpose(1, 2).contiguous()  # [B, 1, HQ, K]

    # Transpose k and v to [B, T, HKV, K/V]
    k = k_rope.permute(0, 2, 1, 3).contiguous()  # [B, T, HKV, K]
    v = v.permute(0, 2, 1, 3).contiguous()  # [B, T, HKV, V]

    return q, k, v


def load_real_data(layer_data_dir: str, layer_idx: int = 0, max_length: int = None, device='cuda'):
    """
    Load real LLM data from dumped files.

    Args:
        layer_data_dir: Directory containing layer_0, layer_1, ... subdirectories
        layer_idx: Which layer to load (default: 0)
        max_length: Maximum sequence length to use (truncate if longer)
        device: Device to load data to

    Returns:
        tuple of (q, k, v) in kernel expected format
    """
    print(f"\n{'='*80}")
    print(f"Loading Real LLM Data from Dumped Files")
    print(f"{'='*80}")
    print(f"Layer data directory: {layer_data_dir}")
    print(f"Layer index: {layer_idx}")
    if max_length:
        print(f"Max length: {max_length}")

    # Load using existing utility
    data_iter = load_qkvh(
        layer_data_dir,
        device=device,
        start_layer=layer_idx,
        max_length=max_length
    )

    layer_data = next(data_iter)
    q_rope = layer_data['q_rope']  # [B, HQ, T, K]
    k_rope = layer_data['k_rope']  # [B, HKV, T, K]
    v = layer_data['v']  # [B, HKV, T, V]

    print(f"\nLoaded data shapes (original layout):")
    print(f"  q_rope: {q_rope.shape}")
    print(f"  k_rope: {k_rope.shape}")
    print(f"  v: {v.shape}")

    # Convert to kernel expected layout
    q, k, v = convert_layout(q_rope, k_rope, v)

    print(f"\nConverted to kernel layout:")
    print(f"  q: {q.shape}  (decode query)")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"\nSequence length: {k.shape[1]}")

    return q, k, v


def benchmark_kernel(
    q, k_q, k_scale, k_zero, k_residual, v,
    BS, SBS, delta,
    precomputed_threshold=None,
    warmup=20, iters=100
):
    """Benchmark a specific configuration."""
    # Warmup
    for _ in range(warmup):
        _ = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v,
            k_residual=k_residual,
            BS=BS, SBS=SBS, delta=delta,
            use_fp8_residual=True,
            precomputed_threshold=precomputed_threshold,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = attn_forward_decode_quantized(
            q=q, k_q=k_q, k_scale=k_scale, k_zero=k_zero, v=v,
            k_residual=k_residual,
            BS=BS, SBS=SBS, delta=delta,
            use_fp8_residual=True,
            precomputed_threshold=precomputed_threshold,
            return_skip_ratio=True,
        )
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    skip_ratio = out[1] if isinstance(out, tuple) else None

    return elapsed_ms, skip_ratio


def benchmark_flash_attn(q, k, v, warmup=20, iters=100):
    """Benchmark FlashAttention baseline."""
    B, _, HQ, K = q.shape
    _, T, HKV, _ = k.shape

    # For GQA, expand k and v to match query heads
    num_repeats = HQ // HKV
    k_expanded = k.repeat_interleave(num_repeats, dim=2)  # [B, T, HQ, K]
    v_expanded = v.repeat_interleave(num_repeats, dim=2)  # [B, T, HQ, V]

    # Prepare inputs for flash_attn_compute
    q_flash = q.squeeze(1)  # [B, HQ, K]

    # Warmup
    for _ in range(warmup):
        _ = flash_attn_compute(q_flash, k_expanded, v_expanded)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = flash_attn_compute(q_flash, k_expanded, v_expanded)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


def test_real_data_optimizations(q, k, v, warmup=20, iters=100):
    """Test optimizations using real LLM data."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK WITH REAL LLM DATA")
    print(f"{'='*80}")
    print(f"Sequence Length: {k.shape[1]}")
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print(f"{'='*80}")

    # Quantize K
    print("\n[Quantizing K to 2-bit + FP8 residual...]")
    k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)
    print(f"  k_q: {k_q.shape}, k_scale: {k_scale.shape}, k_residual: {k_residual.shape}")

    # FlashAttn baseline
    print(f"\n[Baseline] FlashAttention...")
    flash_time = benchmark_flash_attn(q, k, v, warmup=warmup, iters=iters)
    print(f"  Time: {flash_time:.4f} ms")

    # Test configurations
    configs = [
        {"BS": 128, "SBS": 128, "delta": 5.0, "name": "Original (BS=128, delta=5.0)"},
        {"BS": 256, "SBS": 256, "delta": 5.0, "name": "BS=256, SBS=256, delta=5.0"},
        {"BS": 512, "SBS": 256, "delta": 5.0, "name": "BS=512, SBS=256, delta=5.0"},
        {"BS": 512, "SBS": 256, "delta": 6.5, "name": "BS=512, SBS=256, delta=6.5"},
        {"BS": 512, "SBS": 256, "delta": 7.0, "name": "BS=512, SBS=256, delta=7.0"},
    ]

    results = []
    for config in configs:
        print(f"\n[{config['name']}]")
        try:
            time_ms, skip_ratio = benchmark_kernel(
                q, k_q, k_scale, k_zero, k_residual, v,
                BS=config['BS'], SBS=config['SBS'], delta=config['delta'],
                warmup=warmup, iters=iters
            )
            speedup_vs_flash = flash_time / time_ms
            speedup_vs_original = results[0]['time'] / time_ms if results else 1.0

            print(f"  BS={config['BS']}, SBS={config['SBS']}, delta={config['delta']}")
            print(f"  Time: {time_ms:.4f} ms")
            print(f"  Skip ratio: {skip_ratio*100:.2f}%")
            print(f"  Speedup vs FlashAttn: {speedup_vs_flash:.3f}x")
            if results:
                print(f"  Speedup vs Original: {speedup_vs_original:.3f}x")

            results.append({
                "name": config['name'],
                "BS": config['BS'],
                "SBS": config['SBS'],
                "delta": config['delta'],
                "time": time_ms,
                "skip_ratio": skip_ratio,
                "speedup_vs_flash": speedup_vs_flash,
                "speedup_vs_original": speedup_vs_original,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": config['name'],
                "BS": config['BS'],
                "SBS": config['SBS'],
                "delta": config['delta'],
                "error": str(e),
            })

    return results, flash_time


def main():
    parser = argparse.ArgumentParser(description="Test H100 optimizations with real LLM data")
    parser.add_argument("--layer-data-dir", type=str,
                       default="/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data",
                       help="Path to layer data directory")
    parser.add_argument("--layer", type=int, default=0, help="Which layer to test")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence length to use")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()

    print("="*80)
    print(f"H100 Real Data Optimization Tests")
    print("="*80)
    print(f"GPU: {gpu_name}")
    print(f"Layer data directory: {args.layer_data_dir}")
    print(f"Layer: {args.layer}")
    if args.max_length:
        print(f"Max length: {args.max_length}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print("="*80)

    # Load real Q, K, V from dumped data
    q, k, v = load_real_data(
        args.layer_data_dir,
        layer_idx=args.layer,
        max_length=args.max_length,
        device=args.device
    )

    # Run benchmarks
    results, flash_time = test_real_data_optimizations(
        q, k, v,
        warmup=args.warmup,
        iters=args.iters
    )

    # Save results
    all_results = {
        'gpu': gpu_name,
        'layer_data_dir': args.layer_data_dir,
        'layer': args.layer,
        'seq_len': k.shape[1],  # Actual sequence length
        'flash_baseline': flash_time,
        'results': results,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

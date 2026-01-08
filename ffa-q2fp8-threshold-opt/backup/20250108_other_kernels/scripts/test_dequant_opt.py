#!/usr/bin/env python3
"""
Test: Dequant-once optimization vs baseline
Compare performance when dequanting entire BS block once instead of per sub-block
"""
import torch
import time
import sys
sys.path.insert(0, '.')

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_reuse_bs_cudagraph import CUDAGraphDecodeRunnerQ2FP8ReuseBS
from attn_kernel.attn_kernel_opt_dequant_once_cudagraph import CUDAGraphDecodeRunnerOptDequantOnce

def setup_inputs(B, HQ, HKV, K, V, T, device='cuda'):
    """Create test inputs"""
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device=device)  # Decode: Tq=1

    # K quantized to 2-bit
    K_PACKED = K // 4
    k_q = torch.randint(0, 256, (B, T, HKV, K_PACKED), dtype=torch.uint8, device=device)
    k_scale = torch.randn(B, HKV, K, dtype=torch.float16, device=device) * 0.1
    k_zero = torch.randn(B, HKV, K, dtype=torch.float16, device=device) * 0.1

    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device=device)

    return q, k_q, k_scale, k_zero, v

def benchmark_kernel(runner, q, k_q, k_scale, k_zero, v, num_iters=100, warmup=10):
    """Benchmark kernel with warmup"""
    # Warmup
    for _ in range(warmup):
        out = runner.replay(q, k_q, k_scale, k_zero, v)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        out = runner.replay(q, k_q, k_scale, k_zero, v)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / num_iters * 1000
    return out, avg_ms

def test_correctness(out1, out2, tol=1e-3):
    """Check if outputs are close"""
    max_diff = (out1 - out2).abs().max().item()
    mean_diff = (out1 - out2).abs().mean().item()
    return max_diff, mean_diff, max_diff < tol

def main():
    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test configurations
    configs = [
        # (T, BS, SBS, NSB, name)
        (10240, 128, 128, 1, "Short (BS=SBS=128, NSB=1) - No dequant saving"),
        (10240, 256, 128, 2, "Short (BS=256, SBS=128, NSB=2) - 2x dequant saving"),
        (51200, 256, 128, 2, "Medium (BS=256, SBS=128, NSB=2)"),
        (262144, 256, 128, 2, "Long (BS=256, SBS=128, NSB=2)"),
    ]

    B, HQ, HKV, K, V = 1, 24, 8, 128, 128

    for T, BS, SBS, NSB, name in configs:
        print(f"=" * 70)
        print(f"Testing: {name}")
        print(f"  T={T}, BS={BS}, SBS={SBS}, NSB={NSB}")
        print(f"  Dequant reduction: {NSB}x (from {NSB} times to 1 time per block)")
        print(f"=" * 70)

        q, k_q, k_scale, k_zero, v = setup_inputs(B, HQ, HKV, K, V, T, device)

        try:
            # Baseline
            print("\n[1/2] Setting up Baseline kernel...")
            baseline_runner = CUDAGraphDecodeRunnerQ2FP8ReuseBS(
                q, k_q, k_scale, k_zero, v,
                scale=1.0 / (K ** 0.5),
                delta=5.0,
                BS=BS,
                SBS=SBS,
                use_fp8_residual=False,
            )

            out_baseline, time_baseline = benchmark_kernel(baseline_runner, q, k_q, k_scale, k_zero, v, num_iters=50)
            print(f"  Baseline: {time_baseline:.3f} ms")

            # Optimized
            print("\n[2/2] Setting up Optimized kernel (dequant-once)...")
            opt_runner = CUDAGraphDecodeRunnerOptDequantOnce(
                q, k_q, k_scale, k_zero, v,
                scale=1.0 / (K ** 0.5),
                delta=5.0,
                BS=BS,
                SBS=SBS,
                use_fp8_residual=False,
            )

            out_opt, time_opt = benchmark_kernel(opt_runner, q, k_q, k_scale, k_zero, v, num_iters=50)
            print(f"  Optimized: {time_opt:.3f} ms")

            # Compare
            speedup = time_baseline / time_opt
            saving = time_baseline - time_opt
            print(f"\n  Speedup: {speedup:.2f}x")
            print(f"  Time saved: {saving:.3f} ms")

            if speedup > 1.0:
                print(f"  ✓ FASTER by {(speedup-1)*100:.1f}%")
            elif speedup < 1.0:
                print(f"  ✗ SLOWER by {(1-speedup)*100:.1f}%")
            else:
                print(f"  = Same performance")

            # Correctness
            max_diff, mean_diff, passed = test_correctness(out_baseline, out_opt, tol=1e-2)
            print(f"\n  Correctness: {'PASS' if passed else 'FAIL'}")
            print(f"    Max diff:  {max_diff:.6f}")
            print(f"    Mean diff: {mean_diff:.6f}")

            if not passed:
                print("    WARNING: Outputs differ significantly!")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("Optimization: Dequant entire BS block once, reuse across NSB sub-blocks")
    print("Expected speedup: ~5-10% when NSB=2, more when NSB is larger")
    print("Trade-off: Increased register/SRAM usage (stores entire BS block)")

if __name__ == '__main__':
    main()

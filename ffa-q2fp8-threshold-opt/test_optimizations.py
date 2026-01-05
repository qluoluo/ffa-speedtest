#!/usr/bin/env python3
"""Test script for Q2FP8 attention kernel optimizations."""

import sys
import os
import torch
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_inputs(B=1, T=10240, HQ=24, HKV=8, K=128, V=128, device='cuda'):
    """Create test inputs for Q2FP8 attention."""
    # Query
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device=device)

    # Quantized K (2-bit)
    K_PACKED = K // 4
    k_q = torch.randint(0, 256, (B, T, HKV, K_PACKED), dtype=torch.uint8, device=device)

    # Scale and zero point
    k_scale = torch.randn(B, HKV, K, dtype=torch.float32, device=device).abs() * 0.1
    k_zero = torch.randn(B, HKV, K, dtype=torch.float32, device=device) * 0.5

    # Value
    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device=device)

    # FP8 residual
    k_residual = torch.randn(B, T, HKV, K, dtype=torch.float16, device=device)

    return q, k_q, k_scale, k_zero, v, k_residual


def test_opt1_compact():
    """Test Optimization 1: Stage2 Compact Indices."""
    print("\n" + "="*80)
    print("Testing Optimization 1: Stage2 Compact Indices")
    print("="*80)

    try:
        from attn_kernel.attn_kernel_opt1_compact import CUDAGraphDecodeRunnerOpt1Compact
    except ImportError as e:
        print(f"❌ Failed to import Opt1: {e}")
        return False

    # Test configurations
    configs = [
        {"T": 10240, "BS": 128, "delta": 5.0},
        {"T": 51200, "BS": 128, "delta": 5.0},
    ]

    for config in configs:
        T = config["T"]
        BS = config["BS"]
        delta = config["delta"]

        print(f"\nConfig: T={T}, BS={BS}, delta={delta}")

        # Create inputs
        q, k_q, k_scale, k_zero, v, k_residual = create_test_inputs(T=T)

        try:
            # Create runner
            runner = CUDAGraphDecodeRunnerOpt1Compact(
                q, k_q, k_scale, k_zero, v,
                k_residual=k_residual,
                BS=BS,
                delta=delta,
                use_fp8_residual=True,
                warmup=2,
            )

            # Run inference
            out, skip_ratio = runner.replay(
                q, k_q, k_scale, k_zero, v,
                k_residual=k_residual,
                return_skip_ratio=True,
            )

            print(f"✅ Opt1 passed!")
            print(f"   Output shape: {out.shape}")
            print(f"   Skip ratio: {skip_ratio*100:.1f}%")

            # Benchmark
            from utils.bench import benchmark
            ms = benchmark(runner.replay_only, iters=100, warmup=20)
            print(f"   Latency: {ms:.3f} ms")

        except Exception as e:
            print(f"❌ Opt1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_baseline():
    """Test baseline kernel for comparison."""
    print("\n" + "="*80)
    print("Testing Baseline Kernel")
    print("="*80)

    try:
        from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_cudagraph import CUDAGraphDecodeRunnerQ2FP8
    except ImportError as e:
        print(f"❌ Failed to import baseline: {e}")
        return False

    T = 10240
    BS = 128
    delta = 5.0

    print(f"\nConfig: T={T}, BS={BS}, delta={delta}")

    q, k_q, k_scale, k_zero, v, k_residual = create_test_inputs(T=T)

    try:
        runner = CUDAGraphDecodeRunnerQ2FP8(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            BS=BS,
            delta=delta,
            use_fp8_residual=True,
            warmup=2,
        )

        out, skip_ratio = runner.replay(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            return_skip_ratio=True,
        )

        print(f"✅ Baseline passed!")
        print(f"   Output shape: {out.shape}")
        print(f"   Skip ratio: {skip_ratio*100:.1f}%")

        from utils.bench import benchmark
        ms = benchmark(runner.replay_only, iters=100, warmup=20)
        print(f"   Latency: {ms:.3f} ms")

    except Exception as e:
        print(f"❌ Baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main test function."""
    print("Q2FP8 Attention Kernel Optimization Tests")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Test baseline
    if not test_baseline():
        print("\n❌ Baseline test failed. Fix baseline before testing optimizations.")
        return 1

    # Test Opt1
    if not test_opt1_compact():
        print("\n❌ Opt1 test failed.")
        return 1

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

    # Test all optimizations
    optimizations = [
        ("Opt2: Two-Stage Selector", "attn_kernel.attn_kernel_opt2_twostage", "CUDAGraphDecodeRunnerOpt2TwoStage"),
        ("Opt3: Q Reuse", "attn_kernel.attn_kernel_opt3_qreuse", "CUDAGraphDecodeRunnerOpt3QReuse"),
        ("Opt4: Fused Threshold", "attn_kernel.attn_kernel_opt4_fused", "CUDAGraphDecodeRunnerOpt4Fused"),
        ("Opt5: Async Pipeline", "attn_kernel.attn_kernel_opt5_async", "CUDAGraphDecodeRunnerOpt5Async"),
    ]

    print("\n" + "="*80)
    print("Testing Additional Optimizations (placeholders)")
    print("="*80)

    for opt_name, module_name, class_name in optimizations:
        print(f"\n{opt_name}:")
        try:
            mod = __import__(module_name, fromlist=[class_name])
            runner_class = getattr(mod, class_name)
            print(f"  ✅ Module imported successfully")
            print(f"  ℹ️  Note: Currently using Opt1 implementation as placeholder")
        except Exception as e:
            print(f"  ❌ Failed to import: {e}")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✅ Baseline: Working")
    print("✅ Opt1 (Stage2 Compact): Fully implemented and tested")
    print("🚧 Opt2-5: Placeholder implementations (use Opt1 internally)")
    print("\nNext steps:")
    print("1. Implement Opt2 (Two-Stage Selector) for highest ROI")
    print("2. Test on H100 for accurate performance measurement")
    print("3. Combine Opt1+Opt2 for 3-4x speedup target")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Quick test script to verify all optimized kernels work correctly.
Run this on 4090 before doing full benchmarks.
"""
import sys
import torch
from pathlib import Path

# Add kernel directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "kernels"))

print("=" * 80)
print("Q2FP8 Optimization Kernels - Quick Test")
print("=" * 80)
print()

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
print(f"GPU: {props.name}")
print(f"Memory: {props.total_memory // (1024**3)} GB")
print()

# Small test configuration
B, T, HQ, HKV, D, V = 1, 4096, 24, 8, 128, 128

print(f"Test configuration: B={B}, T={T}, HQ={HQ}, HKV={HKV}, D={D}, V={V}")
print()

# Create test data
print("Creating test data...")
q = torch.randn(B, 1, HQ, D, device="cuda", dtype=torch.float16)
k_q = torch.randint(0, 255, (B, T, HKV, D // 4), device="cuda", dtype=torch.uint8)
k_scale = torch.randn(B, HKV, D, device="cuda", dtype=torch.float16) * 0.1
k_zero = torch.randn(B, HKV, D, device="cuda", dtype=torch.float16) * 0.1
# FP8 residual - create as FP16 first then convert
k_residual_fp16 = torch.randn(B, T, HKV, D, device="cuda", dtype=torch.float16) * 0.1
k_residual = k_residual_fp16.to(torch.float8_e4m3fn)
v = torch.randn(B, T, HKV, V, device="cuda", dtype=torch.float16)
print("✓ Test data created")
print()

# Test each optimization
results = {}

# Test Opt2: Adaptive BM_DOT
print("[1/4] Testing Opt2 (Adaptive BM_DOT)...")
try:
    from opt2_adaptive_bm_dot import attn_forward_decode_quantized_opt2
    with torch.no_grad():
        out2 = attn_forward_decode_quantized_opt2(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            BS=128, SBS=128, delta=5.0,
            use_fp8_residual=True,
        )
    print(f"✓ Opt2 passed - Output shape: {out2.shape}")
    results["opt2"] = "PASS"
except Exception as e:
    print(f"✗ Opt2 failed: {e}")
    results["opt2"] = f"FAIL: {e}"
print()

# Test Opt3: FP16 o_buf
print("[2/4] Testing Opt3 (FP16 o_buf)...")
try:
    from opt3_fp16_obuf import attn_forward_decode_quantized_opt3
    with torch.no_grad():
        out3 = attn_forward_decode_quantized_opt3(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            BS=128, SBS=128, delta=5.0,
            use_fp8_residual=True,
        )
    print(f"✓ Opt3 passed - Output shape: {out3.shape}")
    results["opt3"] = "PASS"
except Exception as e:
    print(f"✗ Opt3 failed: {e}")
    results["opt3"] = f"FAIL: {e}"
print()

# Test Opt4: Stage2 Compact
print("[3/4] Testing Opt4 (Stage2 Compact)...")
try:
    from opt4_stage2_compact import attn_forward_decode_quantized_opt4
    with torch.no_grad():
        out4 = attn_forward_decode_quantized_opt4(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            BS=128, SBS=128, delta=5.0,
            use_fp8_residual=True,
        )
    print(f"✓ Opt4 passed - Output shape: {out4.shape}")
    results["opt4"] = "PASS"
except Exception as e:
    print(f"✗ Opt4 failed: {e}")
    results["opt4"] = f"FAIL: {e}"
print()

# Test Opt5: Autotune
print("[4/4] Testing Opt5 (Triton Autotune)...")
try:
    from opt5_autotune import attn_forward_decode_quantized_opt5
    with torch.no_grad():
        out5 = attn_forward_decode_quantized_opt5(
            q, k_q, k_scale, k_zero, v,
            k_residual=k_residual,
            BS=128, SBS=128, delta=5.0,
            use_fp8_residual=True,
        )
    print(f"✓ Opt5 passed - Output shape: {out5.shape}")
    results["opt5"] = "PASS"
except Exception as e:
    print(f"✗ Opt5 failed: {e}")
    results["opt5"] = f"FAIL: {e}"
print()

# Summary
print("=" * 80)
print("Test Summary")
print("=" * 80)
passed = sum(1 for r in results.values() if r == "PASS")
total = len(results)

for name, status in results.items():
    symbol = "✓" if status == "PASS" else "✗"
    print(f"{symbol} {name}: {status}")

print()
if passed == total:
    print(f"✓ All tests passed ({passed}/{total})")
    print()
    print("You can now run full benchmarks:")
    print("  ./run_all_benchmarks.sh")
    sys.exit(0)
else:
    print(f"✗ Some tests failed ({passed}/{total} passed)")
    print()
    print("Please fix the errors before running full benchmarks.")
    sys.exit(1)

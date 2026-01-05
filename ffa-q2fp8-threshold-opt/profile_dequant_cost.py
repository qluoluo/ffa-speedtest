#!/usr/bin/env python3
"""
Profile: What's the actual cost of dequant?
Measure different operations separately
"""
import torch
import triton
import triton.language as tl
import time

@triton.jit
def dequant_only_kernel(
    k_q_in, k_scale, k_zp, k_out,
    K: tl.constexpr, T: tl.constexpr, K_PACKED: tl.constexpr,
):
    """Only dequant, no other operations"""
    pid = tl.program_id(0)

    offs_k = tl.arange(0, K)
    pack_idx = offs_k // 4
    pack_shifts = (offs_k % 4) * 2

    # Load quantized
    kq_ptr = k_q_in + pid * K_PACKED + pack_idx
    kq_tile = tl.load(kq_ptr).to(tl.int32)

    # Unpack
    kq_tile = ((kq_tile >> pack_shifts) & 3).to(tl.float32)

    # Load scale/zp
    scale_ptr = k_scale + offs_k
    zp_ptr = k_zp + offs_k
    scale_tile = tl.load(scale_ptr).to(tl.float32)
    zp_tile = tl.load(zp_ptr).to(tl.float32)

    # Dequant
    k_tile = (kq_tile * scale_tile + zp_tile).to(tl.float16)

    # Store
    k_out_ptr = k_out + pid * K + offs_k
    tl.store(k_out_ptr, k_tile)


@triton.jit
def load_only_kernel(
    k_q_in, k_out,
    K: tl.constexpr, T: tl.constexpr, K_PACKED: tl.constexpr,
):
    """Only load, no dequant"""
    pid = tl.program_id(0)

    offs_k = tl.arange(0, K_PACKED)

    # Load quantized
    kq_ptr = k_q_in + pid * K_PACKED + offs_k
    kq_tile = tl.load(kq_ptr)

    # Store
    k_out_ptr = k_out + pid * K_PACKED + offs_k
    tl.store(k_out_ptr, kq_tile)


@triton.jit
def load_unpack_kernel(
    k_q_in, k_out,
    K: tl.constexpr, T: tl.constexpr, K_PACKED: tl.constexpr,
):
    """Load + unpack, no dequant"""
    pid = tl.program_id(0)

    offs_k = tl.arange(0, K)
    pack_idx = offs_k // 4
    pack_shifts = (offs_k % 4) * 2

    # Load quantized
    kq_ptr = k_q_in + pid * K_PACKED + pack_idx
    kq_tile = tl.load(kq_ptr).to(tl.int32)

    # Unpack
    kq_tile = ((kq_tile >> pack_shifts) & 3).to(tl.float32)

    # Store
    k_out_ptr = k_out + pid * K + offs_k
    tl.store(k_out_ptr, kq_tile.to(tl.float16))


def benchmark_op(kernel, *args, num_iters=1000, warmup=100):
    """Benchmark a kernel"""
    # Warmup
    for _ in range(warmup):
        kernel(*args)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        kernel(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000
    return elapsed


def main():
    device = 'cuda'
    K, T = 128, 10240
    K_PACKED = K // 4

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: K={K}, T={T}")
    print()

    # Create inputs
    k_q = torch.randint(0, 256, (T, K_PACKED), dtype=torch.uint8, device=device)
    k_scale = torch.randn(K, dtype=torch.float16, device=device) * 0.1
    k_zp = torch.randn(K, dtype=torch.float16, device=device) * 0.1
    k_out_packed = torch.empty((T, K_PACKED), dtype=torch.uint8, device=device)
    k_out_full = torch.empty((T, K), dtype=torch.float16, device=device)

    grid = (T,)

    print("Profiling individual operations...")
    print("-" * 50)

    # 1. Load only
    time_load = benchmark_op(
        load_only_kernel[grid],
        k_q, k_out_packed,
        K, T, K_PACKED
    )
    print(f"1. Load only:           {time_load:.4f} ms")

    # 2. Load + Unpack
    time_load_unpack = benchmark_op(
        load_unpack_kernel[grid],
        k_q, k_out_full,
        K, T, K_PACKED
    )
    print(f"2. Load + Unpack:       {time_load_unpack:.4f} ms")
    unpack_cost = time_load_unpack - time_load
    print(f"   Unpack overhead:     {unpack_cost:.4f} ms ({unpack_cost/time_load_unpack*100:.1f}%)")

    # 3. Full pipeline (Load + Unpack + Dequant)
    time_full = benchmark_op(
        dequant_only_kernel[grid],
        k_q, k_scale, k_zp, k_out_full,
        K, T, K_PACKED
    )
    print(f"3. Load + Unpack + Dequant: {time_full:.4f} ms")
    dequant_cost = time_full - time_load_unpack
    print(f"   Dequant overhead:    {dequant_cost:.4f} ms ({dequant_cost/time_full*100:.1f}%)")

    print()
    print("=" * 50)
    print("Breakdown:")
    print(f"  Load:    {time_load:.4f} ms ({time_load/time_full*100:.1f}%)")
    print(f"  Unpack:  {unpack_cost:.4f} ms ({unpack_cost/time_full*100:.1f}%)")
    print(f"  Dequant: {dequant_cost:.4f} ms ({dequant_cost/time_full*100:.1f}%)")
    print(f"  Total:   {time_full:.4f} ms (100%)")
    print()

    if dequant_cost / time_full > 0.20:
        print("✓ Dequant is SIGNIFICANT (>20%) - worth optimizing")
    elif dequant_cost / time_full > 0.10:
        print("~ Dequant is MODERATE (10-20%) - may be worth optimizing")
    else:
        print("✗ Dequant is MINOR (<10%) - NOT the bottleneck")

if __name__ == '__main__':
    main()

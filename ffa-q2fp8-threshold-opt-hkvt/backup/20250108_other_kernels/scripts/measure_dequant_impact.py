#!/usr/bin/env python3
"""
实验：Dequant是否是瓶颈？
方法：对比不同的BS/SBS配置，看dequant次数对性能的影响
"""
import torch
import time
import sys
sys.path.insert(0, '.')

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_reuse_bs_cudagraph import CUDAGraphDecodeRunnerQ2FP8ReuseBS

def setup_inputs(B, HQ, HKV, K, V, T, device='cuda'):
    q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device=device)
    K_PACKED = K // 4
    k_q = torch.randint(0, 256, (B, T, HKV, K_PACKED), dtype=torch.uint8, device=device)
    k_scale = torch.randn(B, HKV, K, dtype=torch.float16, device=device) * 0.1
    k_zero = torch.randn(B, HKV, K, dtype=torch.float16, device=device) * 0.1
    v = torch.randn(B, T, HKV, V, dtype=torch.float16, device=device)
    return q, k_q, k_scale, k_zero, v

def benchmark(runner, q, k_q, k_scale, k_zero, v, num_iters=100):
    for _ in range(10):
        runner.replay(q, k_q, k_scale, k_zero, v)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        runner.replay(q, k_q, k_scale, k_zero, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters * 1000

def main():
    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    B, HQ, HKV, K, V = 1, 24, 8, 128, 128
    T = 10240

    # 实验设计：
    # Config A: BS=128, SBS=128 → NSB=1, dequant 1次
    # Config B: BS=256, SBS=128 → NSB=2, dequant 2次
    # Config C: BS=256, SBS=256 → NSB=1, dequant 1次
    #
    # 如果dequant是瓶颈：
    #   Config B应该比A和C慢（因为dequant 2次）
    # 如果dequant不是瓶颈：
    #   三个config应该差不多（都受其他因素限制）

    configs = [
        (128, 128, "BS=128, SBS=128 (NSB=1, dequant 1x)"),
        (256, 128, "BS=256, SBS=128 (NSB=2, dequant 2x) ← 如果慢很多，说明dequant是瓶颈"),
        (256, 256, "BS=256, SBS=256 (NSB=1, dequant 1x)"),
    ]

    print(f"Testing T={T}")
    print("=" * 70)

    results = []
    for BS, SBS, desc in configs:
        print(f"\n{desc}")
        try:
            q, k_q, k_scale, k_zero, v = setup_inputs(B, HQ, HKV, K, V, T, device)

            runner = CUDAGraphDecodeRunnerQ2FP8ReuseBS(
                q, k_q, k_scale, k_zero, v,
                scale=1.0 / (K ** 0.5),
                delta=5.0,
                BS=BS,
                SBS=SBS,
                use_fp8_residual=False,
            )

            latency = benchmark(runner, q, k_q, k_scale, k_zero, v)
            print(f"  Latency: {latency:.3f} ms")
            results.append((BS, SBS, latency))

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # 分析结果
    print("\n" + "=" * 70)
    print("分析:")
    print("=" * 70)

    if len(results) >= 2:
        bs128 = next((r for r in results if r[0] == 128 and r[1] == 128), None)
        bs256_sbs128 = next((r for r in results if r[0] == 256 and r[1] == 128), None)
        bs256_sbs256 = next((r for r in results if r[0] == 256 and r[1] == 256), None)

        if bs256_sbs128 and bs128:
            slowdown = bs256_sbs128[2] / bs128[2]
            print(f"\nBS=256/SBS=128 vs BS=128/SBS=128:")
            print(f"  {bs256_sbs128[2]:.3f} ms vs {bs128[2]:.3f} ms = {slowdown:.2f}x")

            if slowdown > 1.15:
                print(f"  ✓ 慢了 {(slowdown-1)*100:.1f}% → Dequant是显著瓶颈")
                print(f"    理由: NSB=2时要dequant两次，明显变慢")
            elif slowdown > 1.05:
                print(f"  ~ 慢了 {(slowdown-1)*100:.1f}% → Dequant有一定影响")
            else:
                print(f"  ✗ 几乎无差异 → Dequant不是瓶颈")
                print(f"    理由: dequant 2x vs 1x没有明显性能差异")

        if bs256_sbs256 and bs256_sbs128:
            speedup = bs256_sbs128[2] / bs256_sbs256[2]
            print(f"\nBS=256/SBS=256 vs BS=256/SBS=128 (both NSB=1 vs NSB=2):")
            print(f"  {bs256_sbs256[2]:.3f} ms vs {bs256_sbs128[2]:.3f} ms = {speedup:.2f}x")
            if speedup > 1.10:
                print(f"  ✓ NSB=1明显更快 → Dequant重复是瓶颈")

    print("\n结论: 根据上述对比，可以判断dequant的实际开销占比")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quick correctness check for fused Q2FP8 quantize kernel.
"""

import argparse
import torch

from q2fp8_cache import quantize_symmetric_blocks
from fused_quantize_triton import quantize_symmetric_blocks_triton


def main():
    parser = argparse.ArgumentParser(description="Test fused quantize kernel against reference")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--HKV", type=int, default=8)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--BS", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This test requires CUDA.")

    torch.manual_seed(args.seed)
    k = torch.randn(args.B, args.T, args.HKV, args.K, device=device, dtype=torch.float16)

    k_q_ref, k_scale_ref, k_res_ref = quantize_symmetric_blocks(
        k, block_size=args.BS, k_bits=2
    )
    k_q_tri, k_scale_tri, k_res_tri = quantize_symmetric_blocks_triton(
        k, block_size=args.BS, k_bits=2
    )

    packed_mismatch = (k_q_ref != k_q_tri).float().mean().item()
    scale_max = (k_scale_ref.float() - k_scale_tri.float()).abs().max().item()
    res_max = (k_res_ref.float() - k_res_tri.float()).abs().max().item()

    print("Fused quantize kernel check")
    print(f"  packed mismatch ratio: {packed_mismatch:.6f}")
    print(f"  scale max abs diff:    {scale_max:.6f}")
    print(f"  residual max abs diff: {res_max:.6f}")

    if packed_mismatch > 0.0 or scale_max > 1e-3:
        raise RuntimeError("Fused quantize results differ from reference.")
    print("✅ Passed")


if __name__ == "__main__":
    main()

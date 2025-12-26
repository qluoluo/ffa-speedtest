# Optimization Notes (NVFP4FP8 Decode Path)

This file consolidates optimization ideas near the NVFP4FP8 decode implementation.

## Decode Kernel (`attn_kernel/attn_kernel_v1210_fused_bsz_nvfp4fp8.py`)
- Scale loads: `k_scale` is loaded once per (B, page, HKV). If scale becomes a bottleneck, consider caching it in shared memory or reusing across SB loops.
- Page size: current implementation assumes page size = BS. If a different page size is desired, consider adding a page-size parameter and mapping token indices to scale indices inside the kernel.
- Residual path: use a compile-time flag to elide residual loads when disabled to reduce bandwidth.
- Packing: keep `k_fp4` layout aligned with vectorized loads; consider widening `K_BITS` or unrolling for larger `K`.

## Fused K-Projection + RoPE + NVFP4/FP8 Encoding (`fused_kproj_rope_nvfp4fp8.py`)
- The current implementation uses a reference (Python) path. A fused Triton kernel could eliminate intermediate buffers and reduce overhead.
- Scale computation: using per-page max-abs is simple but can be noisy; consider running statistics or percentile-based scales for stability.
- For streaming decode, maintain per-page scales incrementally instead of recomputing over the full page.

## Memory Layout
- Keep `k_fp4`, `k_scale`, and `k_residual` in layouts that match the decode kernel to minimize permutations.
- Consider storing `k_scale` in fp16 for bandwidth and converting to fp32 in kernel only when needed.

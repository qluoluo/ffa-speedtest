# Optimization Notes (FP8FP8 Decode Path)

This file consolidates optimization ideas near the FP8FP8 decode implementation
without modifying the original kernels.

## Decode Kernel (`attn_kernel/attn_kernel_v1210_fused_bsz_fp8fp8.py`)
- CUDAGraph-friendly path: preallocate and reuse `o/m/l/o_buf` and `mask_buf`
  (avoid per-call allocations) and expose a "plan/run" style API to reduce
  overhead and enable stable graph capture.
- `mask_buf` zeroing: stage1 only writes mask for kept blocks; a variant that
  writes 0/1 for all blocks would allow `torch.empty` instead of `zeros`.
- Stage2 reduction: `for tb in range(0, NTBS)` is fully serial per head; consider
  a two-level reduction (partial m/l/o -> merge) or make `NTBS` a constexpr and
  unroll with `tl.static_range` when T/BS are fixed.
- Threshold path: if `precomputed_threshold` is available, pass it to avoid the
  extra first/last-block dot products inside stage1.
- Kernel tuning: add `triton.autotune` or at least expose `num_warps`,
  `num_stages`, `BM_DOT`, `BS`, `SBS` per-shape to improve occupancy.
- Bandwidth: `o_buf` is fp32; if acceptable, consider fp16/bf16 staging with
  fp32 accumulation in stage2 to reduce memory traffic.

## Fused K-Projection + RoPE + FP8/FP8 Split (`fused_kproj_rope_fp8fp8.py`)
- Incremental update: avoid full-sequence recompute each decode step by
  projecting only the new token(s) and appending to the cache.
- Residual quality: consider bias-corrected residuals or per-head scaling if
  the fp8 base has systematic error.
- Autotune: if a Triton fused path is added, expose `block_t`, `block_h`,
  `num_warps`, `num_stages` for different shapes and GPUs.
- Memory layout: keep `k_fp8/k_residual` in a layout that matches decode
  kernel access patterns to reduce cache misses.

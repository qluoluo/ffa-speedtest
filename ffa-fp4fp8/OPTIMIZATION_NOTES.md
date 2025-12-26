# Optimization Notes (FP4FP8 Decode Path)

This file consolidates optimization ideas near the FP4FP8 decode implementation
without modifying the original kernels.

## Decode Kernel (`attn_kernel/attn_kernel_v1210_fused_bsz_fp4fp8.py`)
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

## Fused K-Projection + RoPE + Quantization (`fused_kproj_rope_fp4fp8.py`)
- Incremental update: avoid full-sequence min/max recompute each decode step by
  maintaining per-(B,HKV,K) running min/max or block-wise scales for new tokens.
- Block-wise quantization: if accuracy allows, compute scales per page/block to
  reduce reduction cost and enable streaming updates.
- Autotune: expose or autotune `block_t`, `block_h`, `block_pair`,
  `num_warps`, `num_stages` for different shapes and GPUs.
- Memory layout: keep `k_scale/k_zero/k_residual` in a layout that matches
  decode kernel access patterns to reduce cache misses.

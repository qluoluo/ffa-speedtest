# Optimization Notes (Q2FP8 Decode Path)

This file tracks optimization ideas for the Q2FP8 decode implementation.
The H100 variant applies several of these changes directly in this folder.

## Decode Kernel (`attn_kernel/attn_kernel_v1210_fused_bsz_q2fp8.py`)
- Implemented: reusable workspace (`Q2FP8DecodeWorkspace`) to preallocate
  `o/m/l/o_buf` + `mask_buf`, avoiding per-call allocations and enabling
  stable CUDAGraph capture.
- Implemented: stage1 writes 0/1 for all blocks, so `mask_buf` can use
  `torch.empty` instead of zeroing.
- Implemented: blocked stage2 reduction to reduce serial work per head.
- Implemented: `triton.autotune` configs for stage1 (BM_DOT/warps/stages) and
  stage2 (reduce block/warps/stages) on H100.
- Remaining: if `precomputed_threshold` is available, pass it to avoid the
  extra first/last-block dot products inside stage1.
- Remaining: consider fp16/bf16 staging for `o_buf` if accuracy allows.

## Fused K-Projection + RoPE + Quantization (`fused_kproj_rope_q2fp8.py`)
- Incremental update: avoid full-sequence min/max recompute each decode step by
  maintaining per-(B,HKV,K) running min/max or block-wise scales for new tokens.
- Block-wise quantization: if accuracy allows, compute scales per page/block to
  reduce reduction cost and enable streaming updates.
- Autotune: expose or autotune `block_t`, `block_h`, `block_pair`,
  `num_warps`, `num_stages` for different shapes and GPUs.
- Memory layout: keep `k_scale/k_zero/k_residual` in a layout that matches
  decode kernel access patterns to reduce cache misses.

# FP8FP8 Decode Acceleration (ffa-fp8fp8)

## Overview
This directory implements a custom decode-time attention path that stores K in FP8 with an FP8 residual and prunes blocks using a threshold heuristic. It also includes a fused K-projection + RoPE path and benchmarking utilities.

## Core Data Layouts
- q: [B, 1, HQ, K]
- k_fp8: [B, T, HKV, K] (FP8 base values; fp16 fallback if fp8 is unavailable)
- k_residual: [B, T, HKV, K] (FP8 residuals for refinement; fp16 fallback if fp8 is unavailable)
- v: [B, T, HKV, V]

## Key Components
### 1) Fused K projection + RoPE + FP8/FP8 split
File: `fused_kproj_rope_fp8fp8.py`
- Builds RoPE cache and applies RoPE to K (per token).
- Stores K as FP8 base values plus FP8 residuals:
  - base = K.cast(fp8)
  - residual = K - base
- Includes a reference implementation and a smoke test to validate outputs.

### 2) FP8FP8 decode attention kernel with pruning
File: `attn_kernel/attn_kernel_v1210_fused_bsz_fp8fp8.py`
- Stage 1 kernel (`attn_forward_stage1_fused_threshold_fp8`):
  - Uses FP8 base values to compute per-block scores and a pruning threshold.
  - Skips a block if all heads in the group are below threshold.
  - Optional residual refinement (`USE_FP8_RESIDUAL`) adds FP8 residuals to K before dot products.
  - Stores per-block partial outputs (m, l, o) and a keep mask.
- Stage 2 kernel (`attn_forward_stage2_masked`):
  - Merges kept blocks with log-sum-exp accumulation to produce the final output.
- Python wrapper `attn_forward_decode_fp8fp8` validates shapes/dtypes and exposes:
  - `BS` (block size), `SBS` (sub-block size), `delta` (threshold margin),
    `precomputed_threshold` (optional), and `use_fp8_residual`.
- Returns output [B, HQ, V] and optionally a skip ratio.

### 3) Benchmarking and plotting
File: `run_attn_bench_fp8fp8.py`
- Loads recorded layer data, converts layouts, quantizes K to FP8+FP8 residual,
  and compares latency vs FlashAttention.
- Caches benchmark results under `plot/` and produces speed/skip-ratio curves.

### 4) Utilities
Files: `utils/*.py`
- `bench.py`: CUDA timing helper.
- `load.py`: loads saved Q/K/V/hidden states and supports truncation.
- `cache.py`: JSON cache read/write helpers for benchmark results.
- `plot.py`: plotting speed and skip ratio.
- `flash.py`: calls FlashAttention for baseline.

## Method Summary
The method stores K as FP8 base values plus an FP8 residual to reduce bandwidth while retaining accuracy, and uses a threshold-based block pruning heuristic to skip blocks with low attention scores. The residual improves accuracy for kept blocks during attention computation.

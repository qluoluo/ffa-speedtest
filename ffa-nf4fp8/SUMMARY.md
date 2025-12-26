# NF4FP8 Decode Acceleration (ffa-nf4fp8)

## Overview
This directory implements a custom decode-time attention path that stores K in NF4 (NormalFloat4) with per-channel scales plus an FP8 residual, and prunes blocks using a threshold heuristic. It also includes a fused K-projection + RoPE + NF4/FP8 encoding path and benchmarking utilities.

## Core Data Layouts
- q: [B, 1, HQ, K]
- k_nf4: [B, T, HKV, K_packed] where K_packed = ceil(K / 2) and 2x4-bit NF4 codes are packed per byte
- k_scale: [B, HKV, K] (per-head, per-channel scale coefficients, shared across T)
- k_residual: [B, T, HKV, K] (FP8 residuals for refinement)
- v: [B, T, HKV, V]

## Key Components
### 1) Fused K projection + RoPE + NF4/FP8 encoding
File: `fused_kproj_rope_nf4fp8.py`
- Builds RoPE cache and applies RoPE to K (per token).
- Encoding scheme:
  - Compute per-(B, HKV, K) absmax across sequence length T.
  - scale = max(abs(min), abs(max)).
  - Quantize normalized values to NF4 codebook entries and pack 2x4-bit codes per byte.
  - Residual = K_fp32 - dequant(K_nf4), stored in FP8 (float8_e5m2 if available, otherwise fp16).
- Triton kernels:
  - `_kproj_rope_minmax_kernel`: computes min/max of projected-and-rope-applied K.
  - `_kproj_rope_nf4_kernel`: recomputes K, applies RoPE, encodes, and writes packed NF4 plus residual and scale.
- Includes a reference implementation and a smoke test to validate fused outputs.
- Constraints: head_dim must be even; only k_bits=4.

### 2) NF4 decode attention kernel with pruning
File: `attn_kernel/attn_kernel_v1210_fused_bsz_nf4fp8.py`
- Stage 1 kernel (`attn_forward_stage1_fused_threshold_nf4`):
  - Decodes K from packed NF4 using per-channel scale.
  - Computes per-block max score to derive a pruning threshold.
    - If no external thresholds, it estimates threshold from the first and last blocks and subtracts `delta`.
  - Skips a block if all heads in the group are below threshold.
  - Optional residual refinement (`USE_FP8_RESIDUAL`) adds FP8 residuals to K before dot products.
  - Stores per-block partial outputs (m, l, o) and a keep mask.
- Stage 2 kernel (`attn_forward_stage2_masked`):
  - Merges kept blocks with log-sum-exp accumulation to produce the final output.
- Python wrapper `attn_forward_decode_nf4` validates shapes/dtypes and exposes:
  - `BS` (block size), `SBS` (sub-block size), `delta` (threshold margin),
    `precomputed_threshold` (optional), and `use_fp8_residual`.
- Returns output [B, HQ, V] and optionally a skip ratio.

### 3) Benchmarking and plotting
File: `run_attn_bench_nf4fp8.py`
- Loads recorded layer data, converts layouts, encodes K to NF4+FP8 residual,
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
The method compresses K into NF4 values with per-channel scales plus an FP8 residual to reduce bandwidth, and uses a threshold-based block pruning heuristic to skip blocks with low attention scores. Residuals recover part of the lost precision during attention computation.

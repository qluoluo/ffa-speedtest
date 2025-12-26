# NVFP4FP8 Decode Acceleration (ffa-nvfp4fp8)

## Overview
This directory implements a custom decode-time attention path that stores K in block-scaled NVFP4 with an FP8 residual and prunes blocks using a threshold heuristic. It also includes a fused K-projection + RoPE + NVFP4/FP8 encoding path (reference implementation) and benchmarking utilities.

## Core Data Layouts
- q: [B, 1, HQ, K]
- k_fp4: [B, T, HKV, K_packed] where K_packed = ceil(K / 2) and 2x4-bit FP4 values are packed per byte
- k_scale: [B, NTB, HKV] per-page scale values (NTB = ceil(T / BS); page size = BS)
- k_residual: [B, T, HKV, K] (FP8 residuals for refinement)
- v: [B, T, HKV, V]

## Key Components
### 1) Fused K projection + RoPE + NVFP4/FP8 encoding
File: `fused_kproj_rope_nvfp4fp8.py`
- Builds RoPE cache and applies RoPE to K (per token).
- Encoding scheme:
  - Page size = BS (block size used by the decode kernel).
  - Compute per-page (B, HKV) max-abs to derive scale: scale = max_abs / 6.
  - Encode scaled values to FP4 (E2M1) nibbles and pack 2x4-bit values per byte.
  - Residual = K_fp32 - decode(NVFP4) stored in FP8 (float8_e5m2 if available, otherwise fp16).
- Uses a reference implementation (Triton placeholder) for now.

### 2) NVFP4 decode attention kernel with pruning
File: `attn_kernel/attn_kernel_v1210_fused_bsz_nvfp4fp8.py`
- Stage 1 kernel (`attn_forward_stage1_fused_threshold_nvfp4`):
  - Decodes K from packed FP4 and multiplies by per-page scale.
  - Computes per-block max score to derive a pruning threshold.
  - Skips a block if all heads in the group are below threshold.
  - Optional residual refinement (`USE_FP8_RESIDUAL`) adds FP8 residuals to K before dot products.
  - Stores per-block partial outputs (m, l, o) and a keep mask.
- Stage 2 kernel (`attn_forward_stage2_masked`):
  - Merges kept blocks with log-sum-exp accumulation to produce the final output.
- Python wrapper `attn_forward_decode_nvfp4fp8` validates shapes/dtypes and exposes:
  - `BS` (block size), `SBS` (sub-block size), `delta` (threshold margin),
    `precomputed_threshold` (optional), and `use_fp8_residual`.
- Returns output [B, HQ, V] and optionally a skip ratio.

### 3) Benchmarking and plotting
File: `run_attn_bench_nvfp4fp8.py`
- Loads recorded layer data, converts layouts, encodes K to NVFP4 + FP8 residual,
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
The method compresses K into block-scaled NVFP4 values plus an FP8 residual to reduce bandwidth, and uses a threshold-based block pruning heuristic to skip blocks with low attention scores. Per-page scale factors provide dynamic range for NVFP4, while residuals recover part of the lost precision during attention computation.

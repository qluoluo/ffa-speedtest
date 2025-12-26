# Q2FP8 Decode Acceleration (ffa-q2fp8)

## Overview
This directory implements a custom decode-time attention path that stores K in 2-bit quantized form with an FP8 residual and prunes blocks using a threshold heuristic. It also includes a fused K-projection + RoPE + quantization path and benchmarking utilities.

## Core Data Layouts
- q: [B, 1, HQ, K]
- k_q: [B, T, HKV, K_packed] where K_packed = ceil(K / 4) and 4x2-bit values are packed per byte
- k_scale, k_zero: [B, HKV, K] (per-head, per-channel scale and zero point, shared across T)
- k_residual: [B, T, HKV, K] (FP8 residuals for dequant refinement)
- v: [B, T, HKV, V]

## Key Components
### 1) Fused K projection + RoPE + int2/FP8 quantization
File: `fused_kproj_rope_q2fp8.py`
- Builds RoPE cache and applies RoPE to K (per token).
- Quantization scheme:
  - Compute per-(B, HKV, K) min/max across sequence length T.
  - scale = (max - min) / 3, zero = min.
  - 2-bit quantization (values 0..3) with rounding ties to even.
  - Pack 4 quantized values into one byte.
  - Residual = K_fp32 - dequant(K_q), stored in FP8 (float8_e5m2 if available, otherwise fp16).
- Triton kernels:
  - `_kproj_rope_minmax_kernel`: computes min/max of projected-and-rope-applied K.
  - `_kproj_rope_quant_kernel`: recomputes K, applies RoPE, quantizes, and writes packed int2 plus residual.
- Includes a reference implementation and a smoke test to validate fused outputs.
- Constraints: head_dim must be even; block_pair must be even; only k_bits=2.

### 2) Quantized decode attention kernel with pruning
File: `attn_kernel/attn_kernel_v1210_fused_bsz_q2fp8.py`
- Stage 1 kernel (`attn_forward_stage1_fused_threshold_qbits`):
  - Dequantizes K from packed int2 using k_scale/k_zero.
  - Computes per-block max score to derive a pruning threshold.
    - If no external thresholds, it estimates threshold from the first and last blocks and subtracts `delta`.
  - Skips a block if all heads in the group are below threshold.
  - Optional residual refinement (`USE_FP8_RESIDUAL`) adds FP8 residuals to K before dot products.
  - Stores per-block partial outputs (m, l, o) and a keep mask.
- Stage 2 kernel (`attn_forward_stage2_masked`):
  - Merges kept blocks with log-sum-exp accumulation to produce the final output.
- Python wrapper `attn_forward_decode_quantized` validates shapes/dtypes and exposes:
  - `BS` (block size), `SBS` (sub-block size), `delta` (threshold margin),
    `precomputed_threshold` (optional), and `use_fp8_residual`.
- Returns output [B, HQ, V] and optionally a skip ratio.

### 3) Benchmarking and plotting
File: `run_attn_bench_q2.py`
- Loads recorded layer data, converts layouts, quantizes K to int2+FP8 residual,
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
The method compresses K into 2-bit values plus an FP8 residual to reduce bandwidth, and uses a threshold-based block pruning heuristic to skip blocks with low attention scores. Quantization scales are per-channel (per head, per dimension) and shared across the sequence, and residuals recover part of the lost precision during attention computation.

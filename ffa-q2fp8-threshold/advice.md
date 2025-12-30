# Advice for Q2FP8 Threshold on H100

## Stage timing snapshot
- GPU: NVIDIA H100 80GB HBM3
- Config: T=256k, BS=256, SBS=256, delta=5, bsz=1, fp16
- Replay-only timing: threshold=0.007 ms, stage1=0.204 ms, stage2=0.042 ms, full=0.257 ms
- Skip ratio: 98.35%

## Key observation
Stage1 dominates (~79% of full). Even with high skip ratio, stage1 still dequantizes and computes block maxima for every block, so skip does not reduce that cost.

## Optimization ideas (priority order)
1) Add a cheaper upper-bound prune before dequant+dot in stage1.
   - Precompute per-block K norm/absmax during K quantization or cache update.
   - Use ||q|| * ||k|| (or absmax bound) to early-reject blocks below threshold.
   - Goal: reduce the number of block-level dequant+dot operations.

2) Specialize kernels for small G (here G=HQ/HKV=3).
   - Current BM_DOT=16 wastes rows with row_mask.
   - Provide BM_DOT=4/8 variants and select by G.
   - Autotune num_warps/num_stages/BS/SBS for H100.

3) Reduce buffer bandwidth between stage1 and stage2.
   - Store o_buf in fp16/bf16 with fp32 accumulation in stage2.
   - Consider fusing partial reduction to avoid full o_buf writes.

4) Avoid full NTBS scan in stage2.
   - Build a compact keep list or two-level reduction.
   - Stage2 should iterate kept blocks only, not all blocks.

5) Increase parallelism when possible.
   - Larger bsz or multi-layer batching improves H100 occupancy.

## Next experiments
- Compare BM_DOT=8 vs BM_DOT=16 for G=3.
- Add an upper-bound prune and re-measure stage1 time.
- Measure stage2 with keep-list compaction to confirm benefit.

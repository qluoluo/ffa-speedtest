# Q4FP8 Optimization Summary

## Environment
- GPU: NVIDIA GeForce RTX 4090 (48GB)
- dtype: fp16
- BS=128, SBS=128, delta=5.0
- T=256k (step=262144)
- iters=50, warmup=20, CUDAGraph replay-only, with non-CG baseline
- data: Llama-3_2-3B/longbench_gov_report_48_68_256k layer_1, bsz=1
- script: run_attn_bench_q4fp8_cudagraph.py

## Variants
- attn_q4fp8_sym_mask: baseline mask-based pruning
- attn_q4fp8_sym_compact: compact keep list (kept_indices/kept_counts) + compact stage2
- attn_q4fp8_sym_lr64_mask: BK=64 low-reg tiling + mask-based pruning
- attn_q4fp8_sym_lr64_compact: BK=64 + compact keep list

## Results (T=256k)
| kernel | Q4 (ms) | Q4_CG (ms) | Flash (ms) | Speedup vs Flash (Q4_CG) | Skip ratio |
| --- | --- | --- | --- | --- | --- |
| attn_q4fp8_sym_mask | 0.249 | 0.243 | 1.125 | 4.64x | 0.998901 |
| attn_q4fp8_sym_compact | 0.216 | 0.201 | 1.125 | 5.60x | 0.998901 |
| attn_q4fp8_sym_lr64_mask | 0.325 | 0.319 | 1.126 | 3.53x | 0.998901 |
| attn_q4fp8_sym_lr64_compact | 0.288 | 0.282 | 1.124 | 3.98x | 0.998901 |

## Commands
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_mask --max-length 262144 --step 262144 --iters 50 --warmup 20 --no-plot --with-q2 --force
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_compact --max-length 262144 --step 262144 --iters 50 --warmup 20 --no-plot --with-q2 --force
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_lr64_mask --max-length 262144 --step 262144 --iters 50 --warmup 20 --no-plot --with-q2 --force
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_lr64_compact --max-length 262144 --step 262144 --iters 50 --warmup 20 --no-plot --with-q2 --force

## Notes
- Compact keep list is the fastest variant on this setup.
- BK=64 low-reg variants were slower than the baseline mask path at 256k on 4090.
- Skip ratio was identical across variants for this data/config.
- Vectorized unpack and split-stage kernels from q2new/split were not ported yet.

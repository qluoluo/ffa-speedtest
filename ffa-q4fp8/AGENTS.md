# Repository Guidelines

## Project Structure & Module Organization
- `run_attn_bench_q4fp8_cudagraph.py`: main benchmarking and plotting entrypoint for Q4FP8 CUDAGraph decode.
- `run_q4_variants.sh`: convenience script to sweep the core kernel variants.
- `attn_kernel/`: Triton kernels and CUDAGraph runners (files follow `attn_q4fp8_*` naming).
- `utils/`: helpers for loading layer data, benchmarking, caching, plotting, and misc utilities.
- `OPTIMIZATION_SUMMARY.md`: latest benchmark results and commands; update when performance claims change.

## Build, Test, and Development Commands
- No build step; run scripts directly with Python on a CUDA-capable GPU.
- `python run_attn_bench_q4fp8_cudagraph.py --help` shows available flags and defaults.
- `python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_mask --force` runs a single kernel.
- `./run_q4_variants.sh` runs the four main variants sequentially.
- Use `--no-plot` to skip figure generation, or `--with-q2` to include the non-CUDAGraph baseline.

## Data, Caches, and Outputs
- Benchmarks expect layer data under `../attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data/` with `layer_0/`, `layer_1/`, ... and `*.pt` tensors.
- Cached results and plots are written under `../attn_analysis/result/.../plot/<kernel>_cudagraph/...` with raw JSON in `raw/`. Use `--force` to bypass cache.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and `snake_case` for functions and variables; keep kernel filenames aligned with `attn_q4fp8_*` patterns.
- Prefer small, focused helpers in `utils/`; keep comments short and meaningful (English or Chinese is fine).
- Avoid adding new dependencies without documenting usage in this file or `OPTIMIZATION_SUMMARY.md`.

## Testing Guidelines
- There is no automated test suite. Validate changes by running a representative benchmark (at least one kernel) and checking for correct output shapes, runtime stability, and expected cache or plot artifacts.
- For kernel changes, run with `--with-q2` to compare against the baseline implementation.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style, e.g. `feat: ...`, `chore: ...`, sometimes in Chinese.
- PRs should include a brief change summary, benchmark commands used, and key results (GPU, dtype, BS/SBS, length). Update `OPTIMIZATION_SUMMARY.md` when performance results change.

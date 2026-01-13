# Repository Guidelines

## Project Structure & Module Organization
- `attn_kernel/`: Python Triton-based Q2FP8 attention kernel variants. New kernels follow `attn_q2fp8_<variant>` naming.
- `utils/`: shared benchmarking, caching, plotting, and data-loading helpers.
- `plot/`: generated benchmark outputs; raw JSON lives under `plot/<kernel>_cudagraph/<GPU>/<run>/raw/`.
- Top-level scripts: `run_attn_bench_q2fp8_cudagraph.py` (main benchmark driver), `test_all_kernels.sh` (batch run), `run_all_256k_tests.sh` (curated 256k sweep), `profile_dequant_cost.py` (micro-benchmark).
- `backup/`: archived experiments; avoid editing unless reviving old work.

## Build, Test, and Development Commands
- `python run_attn_bench_q2fp8_cudagraph.py --attn-kernel attn_q2fp8_base_mask --BS 128 --max-length 262144` runs a single-kernel benchmark; plotting and FlashAttention baselines are enabled unless `--no-plot` or `--no-flash` is set.
- `./test_all_kernels.sh --iters 200 --warmup 50` sweeps every kernel in `attn_kernel/`; use `KERNELS=...` or `KERNEL_FILTER=...` to narrow the run.
- `./run_all_256k_tests.sh` runs the standard 256k suite; tune via env vars such as `BS`, `DELTA`, `ITERS`, `WARMUP`, `NO_PLOT`, `NO_FLASH`.
- `python profile_dequant_cost.py` profiles dequantization cost in isolation.

## Coding Style & Naming Conventions
- Python uses 4-space indentation; keep functions and modules in `snake_case`.
- Bash scripts are `bash`-compatible and favor `set -euo pipefail`.
- Kernel filenames and module names should remain `attn_q2fp8_<variant>`; output folders mirror `<kernel>_cudagraph`.

## Testing Guidelines
- There is no unit-test framework; validation is benchmark-driven.
- Use `test_all_kernels.sh` for smoke checks and compare against the FlashAttention baseline from `run_attn_bench_q2fp8_cudagraph.py` when relevant.
- GPU plus CUDA-enabled PyTorch/Triton are required for any run.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative summaries (sometimes `refactor:` or `update` prefixes). Keep messages concise and include kernel/scope (for example, `256k`).
- PRs should include commands run, hardware/GPU details, key metrics or plots, and any data-path changes (see `EXP_ROOT_DIR` in `run_attn_bench_q2fp8_cudagraph.py`).

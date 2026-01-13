# Repository Guidelines

## Project Structure & Module Organization
This repo organizes FFA/quantized attention benchmarks by quantization variant. Top-level directories include:
- `ffa-fp8/`, `ffa-q1fp8/`, `ffa-q2fp8/`, `ffa-q4fp8/`, `ffa-q2fp8-threshold*`, etc.: per-variant kernels, scripts, and utilities.
  - `attn_kernel/`: kernel implementations.
  - `utils/`: shared helpers (loading, benchmarking, plotting).
  - `e2e/`: end-to-end benchmarks and small generation tests.
- `attn_analysis/`: analysis scripts and data outputs.
- `backup/` folders under some variants: archived experiments; do not treat as active sources unless noted.

## Build, Test, and Development Commands
There is no single build system; each variant ships runnable scripts. Typical examples:
- `python ffa-fp8fp8/e2e/bench_llama_fp8fp8.py --compare --mode decode` (baseline vs fp8fp8 decode).
- `python ffa-q2fp8/e2e/bench_llama_q2fp8.py --mode decode` (q2fp8 e2e bench).
- `bash ffa-q2fp8/run_attn_bench_q2_cudagraph.sh` (cudagraph benchmark).
Check each variant's `SUMMARY.md`, `OPTIMIZATION_NOTES.md`, or `command.md` for exact flags.

## Coding Style & Naming Conventions
- Python and shell scripts; follow existing style (4-space indentation, explicit imports).
- Use `snake_case` for functions/variables; keep filenames descriptive, e.g. `run_attn_bench_*.py`.
- Kernel files typically live in `attn_kernel/` and use versioned names like `attn_kernel_v1210_*.py`; keep new kernels consistent with this pattern.

## Testing Guidelines
- No centralized test runner; tests are script-based.
- E2E sanity checks live under `e2e/` (e.g., `test_generate.py`).
- Record GPU model, CUDA version, driver, and key flags when reporting results.

## Commit & Pull Request Guidelines
- Commit messages follow a conventional prefix: `feat:`, `chore:`, `docs:`, `fix:`; summaries are short and may be English or Chinese.
- PRs should include hardware details, exact commands/flags, and before/after metrics. Update `RESULTS.md`/`SUMMARY.md` when adding benchmark data. Avoid committing large raw artifacts; prefer summaries or links.

## Environment & Configuration Tips
- Most scripts require CUDA plus PyTorch, Triton, and FlashAttention. Reduce `--seq-len` or `--batch` when VRAM is limited.

## Agent Notes
- Some subdirectories include their own `AGENTS.md`; follow them when working in those areas.

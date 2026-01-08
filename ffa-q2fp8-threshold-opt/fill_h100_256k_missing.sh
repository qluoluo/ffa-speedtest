#!/bin/bash

# Run missing 256k benchmarks for H100 (lr64_mask, base_compact, sym_mask, sym_lr64_compact).
# Results are saved under plot/**/raw/*.json by the benchmark scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python not found: ${PYTHON}" >&2
    exit 1
fi

BENCH_BASE="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph.py"
BENCH_SYM="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph_symquant.py"

if [[ ! -f "${BENCH_BASE}" ]]; then
    echo "Benchmark script not found: ${BENCH_BASE}" >&2
    exit 1
fi
if [[ ! -f "${BENCH_SYM}" ]]; then
    echo "Benchmark script not found: ${BENCH_SYM}" >&2
    exit 1
fi

if [[ -z "${SKIP_H100_CHECK:-}" ]]; then
    "${PYTHON}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA is not available.", file=sys.stderr)
    sys.exit(1)

name = torch.cuda.get_device_properties(0).name
print(f"Detected GPU: {name}")
if "H100" not in name:
    print("GPU is not H100. Set SKIP_H100_CHECK=1 to override.", file=sys.stderr)
    sys.exit(1)
PY
fi

BS="${BS:-128}"
SBS="${SBS:-$BS}"
DELTA="${DELTA:-5.0}"
LAYER="${LAYER:-1}"
BSZ="${BSZ:-1}"
STEP="${STEP:-4096}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
ITERS="${ITERS:-500}"
WARMUP="${WARMUP:-100}"

COMMON_ARGS=(
    --BS "${BS}"
    --SBS "${SBS}"
    --delta "${DELTA}"
    --layer "${LAYER}"
    --bsz "${BSZ}"
    --step "${STEP}"
    --max-length "${MAX_LENGTH}"
    --iters "${ITERS}"
    --warmup "${WARMUP}"
)

if [[ "${NO_PLOT:-1}" != "0" ]]; then
    COMMON_ARGS+=(--no-plot)
fi
if [[ "${NO_FLASH:-0}" != "0" ]]; then
    COMMON_ARGS+=(--no-flash)
fi

EXTRA_ARGS=("$@")

failures=()

run_bench() {
    local label="$1"
    shift
    echo "=========================================="
    echo "${label}"
    echo "=========================================="
    if ! "$@"; then
        failures+=("${label}")
    fi
    echo ""
}

run_bench "attn_q2fp8_lr64_mask" \
    "${PYTHON}" "${BENCH_BASE}" \
    --attn-kernel attn_q2fp8_lr64_mask \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_base_compact" \
    "${PYTHON}" "${BENCH_BASE}" \
    --attn-kernel attn_q2fp8_base_compact \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_sym_mask (symquant)" \
    "${PYTHON}" "${BENCH_SYM}" \
    --attn-kernel attn_q2fp8_sym_mask \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_sym_lr64_compact (symquant)" \
    "${PYTHON}" "${BENCH_SYM}" \
    --attn-kernel attn_q2fp8_sym_lr64_compact \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

if [[ "${#failures[@]}" -ne 0 ]]; then
    echo "Completed with failures: ${failures[*]}" >&2
    exit 1
fi

echo "All H100 256k benchmarks completed successfully."

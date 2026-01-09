#!/bin/bash

# Run missing 256k benchmarks locally (lr64_mask, base_compact, sym_mask, sym_lr64_compact).
# Results are saved under plot/**/raw/*.json by the benchmark scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python not found: ${PYTHON}" >&2
    exit 1
fi

BENCH_SCRIPT="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph.py"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "Benchmark script not found: ${BENCH_SCRIPT}" >&2
    exit 1
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

if [[ "${FORCE:-1}" != "0" ]]; then
    COMMON_ARGS+=(--force)
fi

if [[ "${NO_PLOT:-0}" != "0" ]]; then
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
    "${PYTHON}" "${BENCH_SCRIPT}" \
    --attn-kernel attn_q2fp8_lr64_mask \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_base_compact" \
    "${PYTHON}" "${BENCH_SCRIPT}" \
    --attn-kernel attn_q2fp8_base_compact \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_sym_mask (symquant)" \
    "${PYTHON}" "${BENCH_SCRIPT}" \
    --attn-kernel attn_q2fp8_sym_mask \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

run_bench "attn_q2fp8_sym_lr64_compact (symquant)" \
    "${PYTHON}" "${BENCH_SCRIPT}" \
    --attn-kernel attn_q2fp8_sym_lr64_compact \
    "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

if [[ "${#failures[@]}" -ne 0 ]]; then
    echo "Completed with failures: ${failures[*]}" >&2
    exit 1
fi

echo "All 256k benchmarks completed successfully."

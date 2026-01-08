#!/usr/bin/env bash
set -euo pipefail

KERNEL_NAME="${KERNEL_NAME:-attn_q2fp8_sym_lr64_compact}"

DTYPE="${DTYPE:-fp16}"
BS="${BS:-256}"
SBS="${SBS:-256}"
DELTA="${DELTA:-5.0}"
LAYER="${LAYER:-1}"
BSZ="${BSZ:-1}"
MAX_LEN="${MAX_LEN:-262144}"
STEP="${STEP:-${MAX_LEN}}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-1}"
CG_WARMUP="${CG_WARMUP:-1}"
FORCE="${FORCE:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="${NCU_SET:-full}"
KERNEL_FILTER="${KERNEL_FILTER:-regex:attn_}"
OUT_DIR="${OUT_DIR:-${PWD}/ncu_reports}"
NCU_OUTPUT="${NCU_OUTPUT:-${OUT_DIR}/${KERNEL_NAME}_T${MAX_LEN}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph.py"

mkdir -p "${OUT_DIR}"

bench_args=(
  "${PYTHON_BIN}" "${BENCH_PY}"
  --attn-kernel "${KERNEL_NAME}"
  --dtype "${DTYPE}"
  --BS "${BS}"
  --SBS "${SBS}"
  --delta "${DELTA}"
  --layer "${LAYER}"
  --bsz "${BSZ}"
  --max-length "${MAX_LEN}"
  --step "${STEP}"
  --iters "${ITERS}"
  --warmup "${WARMUP}"
  --cg-warmup "${CG_WARMUP}"
  --no-flash
  --no-plot
)
if [[ "${FORCE}" == "1" ]]; then
  bench_args+=(--force)
fi

ncu_cmd=(
  "${NCU_BIN}"
  --replay-mode kernel
  --target-processes all
  -o "${NCU_OUTPUT}"
  --set "${NCU_SET}"
  --kernel-name "${KERNEL_FILTER}"
  -f
)

printf "[Info] %q " "${ncu_cmd[@]}" "${bench_args[@]}"
echo
"${ncu_cmd[@]}" "${bench_args[@]}"

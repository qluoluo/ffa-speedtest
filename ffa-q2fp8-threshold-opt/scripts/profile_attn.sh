#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph.py"
PYTHON_BIN="/remote-home1/zgliu/anaconda3/envs/ffa/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
NCU_BIN="$(command -v ncu || true)"
if [[ -z "${NCU_BIN}" ]]; then
  echo "ncu executable not found in PATH." >&2
  exit 1
fi

KERNEL_NAME="attn_q2fp8_sym_lr64_compact"

DTYPE="fp16"
BS="256"
SBS="256"
DELTA="5.0"
LAYER="1"
BSZ="1"
MAX_LEN="262144"
STEP="${MAX_LEN}"
ITERS="5"
WARMUP="1"
CG_WARMUP="1"
FORCE="0"

NCU_SET="full"
KERNEL_FILTER="regex:attn_"
OUT_DIR="${SCRIPT_DIR}/ncu_reports"
NCU_OUTPUT="${OUT_DIR}/${KERNEL_NAME}_T${MAX_LEN}"

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

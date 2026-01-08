#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
BENCH_PY="${REPO_DIR}/run_attn_bench_q2fp8_cudagraph.py"

KERNEL_NAME="attn_kernel_v1210_fused_bsz_q2fp8_cudagraph"
PYTHON_BIN="${PYTHON_BIN:-/remote-home1/zgliu/anaconda3/envs/ffa/bin/python}"
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
NO_CACHE="${NO_CACHE:-1}"

NCU_BIN="${NCU_BIN:-/remote-home1/zgliu/cudas/cuda-12.1/bin/ncu}"
NCU_SET="${NCU_SET:-full}"
KERNEL_FILTER="${KERNEL_FILTER:-regex:attn_}"
OUT_DIR_BASE="${OUT_DIR:-${REPO_DIR}/ncu_reports}"

if [[ ! -f "${BENCH_PY}" ]]; then
  echo "Error: ${BENCH_PY} not found." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: ${PYTHON_BIN} not found or not executable." >&2
  exit 1
fi

if [[ ! -x "${NCU_BIN}" ]]; then
  echo "Error: ${NCU_BIN} not found or not executable." >&2
  exit 1
fi

gpu_name_safe="unknown_gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_query_index="${CUDA_VISIBLE_DEVICES:-0}"
  gpu_query_index="${gpu_query_index%%,*}"
  if [[ -z "${gpu_query_index}" ]]; then
    gpu_query_index="0"
  fi
  gpu_name_raw="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_query_index}" 2>/dev/null | head -n 1)"
  if [[ -n "${gpu_name_raw}" ]]; then
    gpu_name_safe="$(echo "${gpu_name_raw}" | tr '[:space:]/' '_' | tr -cd '[:alnum:]_.-')"
  fi
fi
if [[ -z "${gpu_name_safe}" ]]; then
  gpu_name_safe="unknown_gpu"
fi

OUT_DIR="${OUT_DIR_BASE}/${gpu_name_safe}"
NCU_OUTPUT="${NCU_OUTPUT:-${OUT_DIR}/${KERNEL_NAME}_T${MAX_LEN}}"

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
if [[ "${NO_CACHE}" == "1" ]]; then
  bench_args+=(--no-cache)
fi

log="${NCU_OUTPUT}.log"
cmd=(
  "${NCU_BIN}"
  --replay-mode kernel
  --target-processes all
  -o "${NCU_OUTPUT}"
  --set "${NCU_SET}"
  --kernel-name "${KERNEL_FILTER}"
  -f
)
cmd+=("${bench_args[@]}")

{
  echo "[RunInfo] timestamp: $(date -Is)"
  echo "[RunInfo] host: $(hostname -f 2>/dev/null || hostname)"
  echo "[RunInfo] user: $(id -un)"
  echo "[RunInfo] cwd: ${REPO_DIR}"
  echo "[RunInfo] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
  echo "[RunInfo] python_bin: ${PYTHON_BIN}"
  echo "[RunInfo] python_version: $("${PYTHON_BIN}" --version 2>&1)"
  echo "[RunInfo] ncu_bin: ${NCU_BIN}"
  echo "[RunInfo] ncu_version: $("${NCU_BIN}" --version 2>&1 | tr '\n' ' ')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[RunInfo] nvidia_smi:"
    nvidia-smi -L 2>&1 | sed 's/^/[RunInfo]   /'
  else
    echo "[RunInfo] nvidia_smi: not found"
  fi
  echo "[RunInfo] params: DTYPE=${DTYPE} BS=${BS} SBS=${SBS} DELTA=${DELTA} LAYER=${LAYER} BSZ=${BSZ} MAX_LEN=${MAX_LEN} STEP=${STEP} ITERS=${ITERS} WARMUP=${WARMUP} CG_WARMUP=${CG_WARMUP} NO_CACHE=${NO_CACHE}"
  echo "[RunInfo] bench_cmd:"
  printf "[RunInfo]   %q" "${bench_args[@]}"
  echo
  echo "[RunInfo] ncu_cmd:"
  printf "[RunInfo]   %q" "${cmd[@]}"
  echo
  if command -v git >/dev/null 2>&1 && git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[RunInfo] git_commit: $(git -C "${REPO_DIR}" rev-parse --short HEAD)"
    if [[ -n "$(git -C "${REPO_DIR}" status --porcelain=v1 2>/dev/null)" ]]; then
      echo "[RunInfo] git_status:"
      git -C "${REPO_DIR}" status --porcelain=v1 2>/dev/null | sed 's/^/[RunInfo]   /'
    else
      echo "[RunInfo] git_status: clean"
    fi
  else
    echo "[RunInfo] git_commit: unavailable"
  fi
} >"${log}"

echo "[Info] Profiling ${KERNEL_FILTER} -> ${NCU_OUTPUT}.ncu-rep"
set +e
"${cmd[@]}" >>"${log}" 2>&1
rc=$?
set -e
if [[ "${rc}" -ne 0 ]]; then
  echo "[Error] Nsight Compute failed (rc=${rc}). See ${log}." >&2
  exit "${rc}"
fi
if grep -q "ERR_NVGPUCTRPERM" "${log}"; then
  echo "[Error] Nsight Compute cannot access GPU performance counters (ERR_NVGPUCTRPERM)." >&2
  exit 1
fi
if [[ ! -f "${NCU_OUTPUT}.ncu-rep" ]]; then
  echo "[Error] Nsight Compute did not produce a report. See ${log}." >&2
  exit 1
fi

if [[ -n "${SUDO_USER:-}" ]]; then
  chown -R "${SUDO_USER}:${SUDO_USER}" "${OUT_DIR}" || true
fi

echo "[Done] Nsight Compute report: ${NCU_OUTPUT}.ncu-rep"

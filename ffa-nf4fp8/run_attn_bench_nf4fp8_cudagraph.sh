#!/usr/bin/env sh
set -e

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ffa

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PY_SCRIPT="${SCRIPT_DIR}/run_attn_bench_nf4fp8_cudagraph.py"
BS_LIST="64 128 256 512 1024"
SBS_LIST="64 128 256 512 1024"
ITERS="${ITERS:-100}"

for bs in ${BS_LIST}; do
  for sbs in ${SBS_LIST}; do
    if [ "${bs}" -lt "${sbs}" ]; then
      continue
    fi
    echo "[Info] BS=${bs} SBS=${sbs} ITERS=${ITERS}"
    python "${PY_SCRIPT}" --BS "${bs}" --SBS "${sbs}" --iters "${ITERS}"
  done
done
python /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/occ.py

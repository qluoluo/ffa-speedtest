#!/usr/bin/env sh

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ffa

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PY_SCRIPT="${SCRIPT_DIR}/run_attn_bench_q2_cudagraph.py"
BS_LIST="64 128 256 512 1024 2048 4096"
SBS_LIST="64 128 256 512"
ITERS="100"
DELTA_LIST="5.0 8.0 10.0"

for delta in ${DELTA_LIST}; do
  for bs in ${BS_LIST}; do
    for sbs in ${SBS_LIST}; do
      if [ "${bs}" -lt "${sbs}" ]; then
        continue
      fi
      echo "[Info] BS=${bs} SBS=${sbs} ITERS=${ITERS} DELTA=${delta}"
      python "${PY_SCRIPT}" --BS "${bs}" --SBS "${sbs}" --iters "${ITERS}" --delta "${delta}"
    done
  done
done
python /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/occ.py

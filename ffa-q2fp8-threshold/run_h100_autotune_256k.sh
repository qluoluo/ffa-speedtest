#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BENCH_SCRIPT="${ROOT}/run_attn_bench_q2_cudagraph.py"

# 256k tokens = 262144. Override via env if needed.
MAX_LEN="${MAX_LEN:-262144}"
STEP="${STEP:-$MAX_LEN}"
ITERS="${ITERS:-300}"
WARMUP="${WARMUP:-50}"
CG_WARMUP="${CG_WARMUP:-2}"
LAYER="${LAYER:-1}"
ENABLE_FLASH="${ENABLE_FLASH:-1}"

BS_LIST_STR="${BS_LIST:-128}"
SBS_LIST_STR="${SBS_LIST:-128}"
DELTA_LIST_STR="${DELTA_LIST:-5}"
BSZ_LIST_STR="${BSZ_LIST:-1}"
DTYPE_LIST_STR="${DTYPE_LIST:-fp16}"
NUM_WARPS_LIST_STR="${NUM_WARPS_LIST:-4 8}"
NUM_STAGES_LIST_STR="${NUM_STAGES_LIST:-2 3}"
NUM_WARPS_TH_LIST_STR="${NUM_WARPS_TH_LIST:-$NUM_WARPS_LIST_STR}"
NUM_STAGES_TH_LIST_STR="${NUM_STAGES_TH_LIST:-$NUM_STAGES_LIST_STR}"
NUM_WARPS_S1_LIST_STR="${NUM_WARPS_S1_LIST:-$NUM_WARPS_LIST_STR}"
NUM_STAGES_S1_LIST_STR="${NUM_STAGES_S1_LIST:-$NUM_STAGES_LIST_STR}"
NUM_WARPS_S2_LIST_STR="${NUM_WARPS_S2_LIST:-$NUM_WARPS_LIST_STR}"
NUM_STAGES_S2_LIST_STR="${NUM_STAGES_S2_LIST:-$NUM_STAGES_LIST_STR}"

OUT_CSV="${OUT_CSV:-${ROOT}/plot/h100_sweep_256k.csv}"
LOG_DIR="${LOG_DIR:-${ROOT}/plot/h100_sweep_logs}"

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"

CSV_HEADER="timestamp,BS,SBS,delta,bsz,dtype,num_warps_th,num_stages_th,num_warps_s1,num_stages_s1,num_warps_s2,num_stages_s2,q2_ms,q2_cg_ms,flash_ms,skip_ratio,json_path"
if [[ -f "$OUT_CSV" ]]; then
  existing_header="$(head -n1 "$OUT_CSV")"
  if [[ "$existing_header" != "$CSV_HEADER" ]]; then
    OUT_CSV="${OUT_CSV%.csv}_stage.csv"
  fi
fi
if [[ ! -f "$OUT_CSV" ]]; then
  echo "$CSV_HEADER" > "$OUT_CSV"
fi

read -r -a BS_LIST <<< "$BS_LIST_STR"
read -r -a SBS_LIST <<< "$SBS_LIST_STR"
read -r -a DELTA_LIST <<< "$DELTA_LIST_STR"
read -r -a BSZ_LIST <<< "$BSZ_LIST_STR"
read -r -a DTYPE_LIST <<< "$DTYPE_LIST_STR"
read -r -a NUM_WARPS_TH_LIST <<< "$NUM_WARPS_TH_LIST_STR"
read -r -a NUM_STAGES_TH_LIST <<< "$NUM_STAGES_TH_LIST_STR"
read -r -a NUM_WARPS_S1_LIST <<< "$NUM_WARPS_S1_LIST_STR"
read -r -a NUM_STAGES_S1_LIST <<< "$NUM_STAGES_S1_LIST_STR"
read -r -a NUM_WARPS_S2_LIST <<< "$NUM_WARPS_S2_LIST_STR"
read -r -a NUM_STAGES_S2_LIST <<< "$NUM_STAGES_S2_LIST_STR"

find_json_path() {
  local log_file="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -o "plot/[^ ]+_cudagraph_replay\\.json|plot/[^ ]+_cudagraph\\.json" "$log_file" | tail -n1
  else
    grep -Eo "plot/[^ ]+_cudagraph_replay\\.json|plot/[^ ]+_cudagraph\\.json" "$log_file" | tail -n1
  fi
}

extract_metrics() {
  local json_path="$1"
  "$PYTHON" - <<'PY' "$json_path"
import json
import math
import sys

path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)

def last_value(key):
    vals = data.get(key, [])
    if not vals:
        return float("nan")
    val = vals[-1]
    if val is None:
        return float("nan")
    return float(val)

q2 = last_value("q2_ms")
q2cg = last_value("q2_cg_ms")
flash = last_value("flash_ms")
skip = last_value("skip_ratios")

print(f"{q2},{q2cg},{flash},{skip}")
PY
}

for dtype in "${DTYPE_LIST[@]}"; do
  for bsz in "${BSZ_LIST[@]}"; do
    for BS in "${BS_LIST[@]}"; do
      for SBS in "${SBS_LIST[@]}"; do
        if (( SBS > BS )); then
          continue
        fi
        for delta in "${DELTA_LIST[@]}"; do
          for num_warps_th in "${NUM_WARPS_TH_LIST[@]}"; do
            for num_stages_th in "${NUM_STAGES_TH_LIST[@]}"; do
              for num_warps_s1 in "${NUM_WARPS_S1_LIST[@]}"; do
                for num_stages_s1 in "${NUM_STAGES_S1_LIST[@]}"; do
                  for num_warps_s2 in "${NUM_WARPS_S2_LIST[@]}"; do
                    for num_stages_s2 in "${NUM_STAGES_S2_LIST[@]}"; do
                      ts="$(date +%Y-%m-%dT%H:%M:%S)"
                      log_file="${LOG_DIR}/sweep_${ts}_BS${BS}_SBS${SBS}_delta${delta}_bsz${bsz}_${dtype}_nwT${num_warps_th}nsT${num_stages_th}_nw1${num_warps_s1}ns1${num_stages_s1}_nw2${num_warps_s2}ns2${num_stages_s2}.log"
                      cmd=(
                        "$PYTHON" "$BENCH_SCRIPT"
                        --dtype "$dtype"
                        --BS "$BS"
                        --SBS "$SBS"
                        --delta "$delta"
                        --layer "$LAYER"
                        --bsz "$bsz"
                        --max-length "$MAX_LEN"
                        --step "$STEP"
                        --iters "$ITERS"
                        --warmup "$WARMUP"
                        --cg-warmup "$CG_WARMUP"
                        --num-warps-th "$num_warps_th"
                        --num-stages-th "$num_stages_th"
                        --num-warps-s1 "$num_warps_s1"
                        --num-stages-s1 "$num_stages_s1"
                        --num-warps-s2 "$num_warps_s2"
                        --num-stages-s2 "$num_stages_s2"
                        --no-plot
                      )
                      if [[ "$ENABLE_FLASH" == "0" ]]; then
                        cmd+=(--no-flash)
                      fi

                      echo "[Run] dtype=$dtype bsz=$bsz BS=$BS SBS=$SBS delta=$delta nwT=$num_warps_th nsT=$num_stages_th nw1=$num_warps_s1 ns1=$num_stages_s1 nw2=$num_warps_s2 ns2=$num_stages_s2"
                      "${cmd[@]}" | tee "$log_file"

                      json_rel="$(find_json_path "$log_file")"
                      if [[ -z "$json_rel" ]]; then
                        echo "[Warn] Failed to find JSON path in log: $log_file"
                        continue
                      fi

                      if [[ "$json_rel" = /* ]]; then
                        json_path="$json_rel"
                      else
                        json_path="${ROOT}/${json_rel}"
                      fi

                      metrics="$(extract_metrics "$json_path")"
                      echo "$ts,$BS,$SBS,$delta,$bsz,$dtype,$num_warps_th,$num_stages_th,$num_warps_s1,$num_stages_s1,$num_warps_s2,$num_stages_s2,$metrics,$json_path" >> "$OUT_CSV"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

"$PYTHON" - <<'PY' "$OUT_CSV"
import csv
import math
import sys

path = sys.argv[1]
rows = []
with open(path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            q2cg = float(row["q2_cg_ms"])
        except Exception:
            continue
        if math.isnan(q2cg):
            continue
        rows.append((q2cg, row))

rows.sort(key=lambda x: x[0])
print("Top configs by q2_cg_ms:")
for q2cg, row in rows[:10]:
    print(
        f"q2_cg_ms={q2cg:.6f} "
        f"BS={row['BS']} SBS={row['SBS']} delta={row['delta']} "
        f"nwT={row['num_warps_th']} nsT={row['num_stages_th']} "
        f"nw1={row['num_warps_s1']} ns1={row['num_stages_s1']} "
        f"nw2={row['num_warps_s2']} ns2={row['num_stages_s2']} "
        f"bsz={row['bsz']} dtype={row['dtype']}"
    )
PY

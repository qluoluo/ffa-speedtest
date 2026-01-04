#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Override via env vars if needed.
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
LENGTH="${LENGTH:-262144}"
DELTA="${DELTA:-5.0}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-100}"
CG_WARMUP="${CG_WARMUP:-2}"
BASE_BS="${BASE_BS:-128}"
BASE_SBS="${BASE_SBS:-128}"
BS_LIST="${BS_LIST:-"128 256 512"}"
SBS_LIST="${SBS_LIST:-"128 256 512"}"
REPEATS="${REPEATS:-1}"
REPEATS_SWEEP="${REPEATS_SWEEP:-$REPEATS}"
REPEATS_BASE="${REPEATS_BASE:-$REPEATS}"
REPEATS_NOFP8="${REPEATS_NOFP8:-$REPEATS}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-0}"
AGGREGATE="${AGGREGATE:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/reports/stage_timing_${STAMP}}"

mkdir -p "$OUT_DIR"

run_one() {
  local label="$1"
  local bs="$2"
  local sbs="$3"
  local rep="$4"
  shift 4
  local out_file="${OUT_DIR}/${label}_BS${bs}_SBS${sbs}_len${LENGTH}_r${rep}_${STAMP}.json"

  echo "[Run] ${label} BS=${bs} SBS=${sbs} len=${LENGTH} -> ${out_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT/run_attn_bench_q2_stage_timing_cudagraph_report.py" \
    --length "$LENGTH" \
    --BS "$bs" \
    --SBS "$sbs" \
    --delta "$DELTA" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --cg-warmup "$CG_WARMUP" \
    --force \
    "$@" \
    --out "$out_file"
}

for rep in $(seq 1 "$REPEATS_BASE"); do
  run_one baseline "$BASE_BS" "$BASE_SBS" "$rep"
  if [ "$SLEEP_BETWEEN" -gt 0 ]; then
    sleep "$SLEEP_BETWEEN"
  fi
done

for rep in $(seq 1 "$REPEATS_NOFP8"); do
  run_one nofp8 "$BASE_BS" "$BASE_SBS" "$rep" --no-fp8-residual
  if [ "$SLEEP_BETWEEN" -gt 0 ]; then
    sleep "$SLEEP_BETWEEN"
  fi
done

for bs in $BS_LIST; do
  for sbs in $SBS_LIST; do
    if [ "$sbs" -le "$bs" ]; then
      for rep in $(seq 1 "$REPEATS_SWEEP"); do
        run_one sweep "$bs" "$sbs" "$rep"
        if [ "$SLEEP_BETWEEN" -gt 0 ]; then
          sleep "$SLEEP_BETWEEN"
        fi
      done
    fi
  done
done

if [ "$AGGREGATE" -ne 0 ]; then
  summary_file="${OUT_DIR}/summary_${STAMP}.json"
  python - <<PY
import json
import re
from pathlib import Path
from datetime import datetime, timezone

out_dir = Path("$OUT_DIR")
files = [p for p in out_dir.glob("*.json") if not p.name.startswith("summary_")]

pattern = re.compile(r"^(?P<label>.+)_BS(?P<bs>\\d+)_SBS(?P<sbs>\\d+)_len(?P<len>\\d+)_r(?P<rep>\\d+)_")
groups = {}
for path in files:
    m = pattern.match(path.name)
    if not m:
        continue
    key = (
        m.group("label"),
        int(m.group("bs")),
        int(m.group("sbs")),
        int(m.group("len")),
    )
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    entry = groups.setdefault(key, {"files": [], "metrics": {}})
    entry["files"].append(path.name)
    for k in ("threshold_last", "stage1_last", "stage2_last", "stage_sum_last", "full_last"):
        v = summary.get(k)
        if v is None:
            continue
        entry["metrics"].setdefault(k, []).append(float(v))

def mean(vals):
    return sum(vals) / len(vals) if vals else None

def stdev(vals):
    if len(vals) < 2:
        return None
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5

groups_out = []
for (label, bs, sbs, length), payload in sorted(groups.items()):
    metrics = payload["metrics"]
    mean_metrics = {k: mean(v) for k, v in metrics.items()}
    stdev_metrics = {k: stdev(v) for k, v in metrics.items()}
    groups_out.append({
        "label": label,
        "BS": bs,
        "SBS": sbs,
        "length": length,
        "count": max(len(v) for v in metrics.values()) if metrics else 0,
        "mean": mean_metrics,
        "stdev": stdev_metrics,
        "files": sorted(payload["files"]),
    })

summary = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "out_dir": str(out_dir),
    "groups": groups_out,
}
summary_file = Path("$summary_file")
summary_file.write_text(json.dumps(summary, indent=2))
print(f"[Summary] {summary_file}")
PY
fi

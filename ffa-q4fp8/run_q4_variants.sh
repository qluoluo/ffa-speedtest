#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_mask --force 
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_compact --force 
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_lr64_mask --force 
python run_attn_bench_q4fp8_cudagraph.py --attn-kernel attn_q4fp8_sym_lr64_compact --force 

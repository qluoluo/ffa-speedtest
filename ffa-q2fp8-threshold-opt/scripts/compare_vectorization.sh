#!/bin/bash
# Compare original vs vectorized Stage2 performance

echo "=========================================="
echo "Stage2 Vectorization Performance Test"
echo "=========================================="
echo ""

LAYER=1
BSZ=1
BS=128
SBS=128
DELTA=5.0
STEP=65536
ITERS=300
WARMUP=50
MAX_KEPT_RATIO=0.02  # Use optimized ratio

echo "Testing original kernel (attn_q2fp8_sym_lr64_atomic_compact)..."
python scripts/run_attn_bench_q2fp8_cudagraph.py \
    --attn-kernel attn_q2fp8_sym_lr64_atomic_compact \
    --layer ${LAYER} \
    --bsz ${BSZ} \
    --BS ${BS} \
    --SBS ${SBS} \
    --delta ${DELTA} \
    --step ${STEP} \
    --iters ${ITERS} \
    --warmup ${WARMUP} \
    --cg-replay-only \
    --with-q2 \
    --max-kept-ratio ${MAX_KEPT_RATIO} \
    --force

echo ""
echo "=========================================="
echo ""

echo "Testing vectorized kernel (attn_q2fp8_sym_lr64_atomic_compact_vec)..."
python scripts/run_attn_bench_q2fp8_cudagraph.py \
    --attn-kernel attn_q2fp8_sym_lr64_atomic_compact_vec \
    --layer ${LAYER} \
    --bsz ${BSZ} \
    --BS ${BS} \
    --SBS ${SBS} \
    --delta ${DELTA} \
    --step ${STEP} \
    --iters ${ITERS} \
    --warmup ${WARMUP} \
    --cg-replay-only \
    --with-q2 \
    --max-kept-ratio ${MAX_KEPT_RATIO} \
    --force

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="

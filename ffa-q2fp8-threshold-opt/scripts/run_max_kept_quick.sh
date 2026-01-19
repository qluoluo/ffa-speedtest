#!/bin/bash
# Quick sweep of max_kept_ratio values (fewer iterations for faster results)

ATTN_KERNEL="attn_q2fp8_sym_lr64_atomic_compact"
LAYER=1
BSZ=1
BS=128
SBS=128
DELTA=5.0
STEP=65536  # Larger step for fewer points
ITERS=200   # Fewer iterations
WARMUP=50

# Test different max_kept_ratio values
for RATIO in 0.2 0.1 0.05 0.02; do
    echo "=========================================="
    echo "Testing max_kept_ratio=${RATIO}"
    echo "=========================================="

    python scripts/run_attn_bench_q2fp8_cudagraph.py \
        --attn-kernel ${ATTN_KERNEL} \
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
        --max-kept-ratio ${RATIO} \
        --force

    echo ""
done

echo "All benchmarks completed!"
echo "Results saved in plot/${ATTN_KERNEL}_cudagraph/"

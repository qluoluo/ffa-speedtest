#!/bin/bash
# Master script to run all Q2FP8 optimization benchmarks
# Run this on 4090 first to verify, then on H100

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Q2FP8 Optimization Benchmark Suite"
echo "========================================================================"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Is CUDA installed?${NC}"
    exit 1
fi

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo -e "${GREEN}GPU: ${GPU_NAME} (${GPU_MEM} MB)${NC}"
echo ""

# Create results directory with GPU name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Extract simplified GPU name (e.g., "NVIDIA GeForce RTX 4090" -> "RTX4090")
GPU_SHORT=$(echo "${GPU_NAME}" | sed -e 's/NVIDIA //g' -e 's/GeForce //g' -e 's/ //g')
RESULTS_DIR="./results_${GPU_SHORT}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Configuration
SEQ_LEN=262144
BS=256
SBS=256
DELTA=5.0
WARMUP=10
ITERS=100

echo "Configuration:"
echo "  Sequence Length: ${SEQ_LEN}"
echo "  BS: ${BS}, SBS: ${SBS}"
echo "  Delta: ${DELTA}"
echo "  Warmup: ${WARMUP}, Iterations: ${ITERS}"
echo ""

# Find the benchmark script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCH_SCRIPT="${SCRIPT_DIR}/benchmarks/benchmark_all_opts.py"

if [ ! -f "${BENCH_SCRIPT}" ]; then
    echo -e "${RED}ERROR: Benchmark script not found at ${BENCH_SCRIPT}${NC}"
    echo "Please ensure the script is in the correct location."
    exit 1
fi

echo "========================================================================"
echo "Running benchmarks..."
echo "========================================================================"
echo ""

# Run benchmark
python3 "${BENCH_SCRIPT}" \
    --batch-size 1 \
    --seq-len ${SEQ_LEN} \
    --BS ${BS} \
    --SBS ${SBS} \
    --delta ${DELTA} \
    --warmup ${WARMUP} \
    --iters ${ITERS} \
    --output "${RESULTS_DIR}/benchmark_results.json" \
    2>&1 | tee "${RESULTS_DIR}/benchmark_log.txt"

BENCH_STATUS=$?

echo ""
if [ $BENCH_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmarks completed successfully!${NC}"
else
    echo -e "${RED}✗ Benchmarks failed with exit code ${BENCH_STATUS}${NC}"
    exit $BENCH_STATUS
fi

echo ""
echo "========================================================================"
echo "Results Summary"
echo "========================================================================"

# Parse and display results if jq is available
if command -v jq &> /dev/null && [ -f "${RESULTS_DIR}/benchmark_results.json" ]; then
    echo ""
    echo "Speedup Summary:"
    echo "----------------"

    # Extract baseline time
    BASELINE_TIME=$(jq -r '.results.baseline.mean_ms // 0' "${RESULTS_DIR}/benchmark_results.json")

    if [ "$BASELINE_TIME" != "0" ] && [ "$BASELINE_TIME" != "null" ]; then
        echo "Baseline: ${BASELINE_TIME} ms (1.00x)"

        # Calculate speedups for each optimization
        for OPT in opt2_adaptive_bm_dot opt3_fp16_obuf opt4_stage2_compact opt5_autotune; do
            OPT_TIME=$(jq -r ".results.${OPT}.mean_ms // 0" "${RESULTS_DIR}/benchmark_results.json")
            if [ "$OPT_TIME" != "0" ] && [ "$OPT_TIME" != "null" ]; then
                SPEEDUP=$(echo "scale=3; ${BASELINE_TIME} / ${OPT_TIME}" | bc)
                OPT_NAME=$(echo $OPT | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1')
                printf "%-25s: %.4f ms (%.3fx speedup)\n" "$OPT_NAME" "$OPT_TIME" "$SPEEDUP"
            else
                echo "${OPT}: ERROR or not available"
            fi
        done
    else
        echo "Could not parse baseline time from results"
    fi
else
    echo "jq not available for detailed parsing, check ${RESULTS_DIR}/benchmark_results.json manually"
fi

echo ""
echo "========================================================================"
echo "Next Steps"
echo "========================================================================"
echo ""
echo "1. Review results in: ${RESULTS_DIR}/"
echo "2. Check ${RESULTS_DIR}/benchmark_log.txt for detailed output"
echo "3. If running on 4090, copy this directory to H100 and run again"
echo "4. Compare results between GPUs"
echo ""
echo "To run on H100:"
echo "  scp -r $(pwd) user@h100-machine:/path/to/destination/"
echo "  ssh user@h100-machine"
echo "  cd /path/to/destination/"
echo "  ./run_all_benchmarks.sh"
echo ""

exit 0

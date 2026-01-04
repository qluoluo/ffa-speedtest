#!/bin/bash
# Script to run Q2FP8 optimization benchmarks on H100
# Since storage is shared, results will appear in the same directory

set -e

echo "========================================================================"
echo "Q2FP8 Optimization Benchmark - H100 Test Runner"
echo "========================================================================"
echo ""

# Check if running on H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "Detected GPU: ${GPU_NAME}"
echo ""

if [[ ! "${GPU_NAME}" =~ "H100" ]]; then
    echo "WARNING: This script is designed for H100, but detected: ${GPU_NAME}"
    echo "Do you want to continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "Working directory: ${SCRIPT_DIR}"
echo ""

# Check Python environment
echo "Checking Python environment..."
python3 --version
echo ""

# Check CUDA
echo "Checking CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
echo ""

# Check Triton
echo "Checking Triton..."
python3 -c "import triton; print(f'Triton version: {triton.__version__}')" 2>&1 || echo "Triton not found or error"
echo ""

echo "========================================================================"
echo "Starting Benchmark on H100"
echo "========================================================================"
echo ""

# Run the benchmark
./run_all_benchmarks.sh

echo ""
echo "========================================================================"
echo "H100 Benchmark Complete!"
echo "========================================================================"
echo ""
echo "Results are saved in the shared storage at:"
echo "  ${SCRIPT_DIR}/results_*/"
echo ""
echo "You can compare H100 vs RTX4090 results by checking the directory names:"
echo "  - results_H100_*        (H100 results)"
echo "  - results_RTX4090_*     (RTX 4090 results)"
echo ""

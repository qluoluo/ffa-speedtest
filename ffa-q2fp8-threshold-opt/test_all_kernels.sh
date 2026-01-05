#!/bin/bash

# Test all Q2FP8 kernels under attn_kernel/
# This script runs the benchmark for each kernel variant found.
#
# Usage:
#   ./test_all_kernels.sh [additional arguments]
#
# Optional environment variables:
#   KERNELS="k1 k2 k3"        Space- or comma-separated kernel module names to run.
#   KERNEL_FILTER="regex"    Filter auto-discovered kernels with an extended regex.
#
# Examples:
#   ./test_all_kernels.sh --BS 256 --delta 3.0
#   ./test_all_kernels.sh --iters 1000 --warmup 200
#   ./test_all_kernels.sh --layer 2 --bsz 4 --step 2048
#   KERNELS="attn_kernel_opt1_compact,attn_kernel_opt4_fused" ./test_all_kernels.sh --BS 256
#   KERNEL_FILTER="cudagraph" ./test_all_kernels.sh --iters 200

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${SCRIPT_DIR}/attn_kernel"
BENCH_SCRIPT="${SCRIPT_DIR}/run_attn_bench_q2fp8_cudagraph.py"

if [[ ! -d "${KERNEL_DIR}" ]]; then
    echo "Kernel directory not found: ${KERNEL_DIR}"
    exit 1
fi

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "Benchmark script not found: ${BENCH_SCRIPT}"
    exit 1
fi

KERNELS_OVERRIDE="${KERNELS:-}"
KERNEL_FILTER="${KERNEL_FILTER:-}"
if [[ -n "${KERNELS_OVERRIDE}" ]]; then
    KERNELS_OVERRIDE="${KERNELS_OVERRIDE//,/ }"
    read -r -a kernels <<< "${KERNELS_OVERRIDE}"
else
    mapfile -t kernels < <(
        find "${KERNEL_DIR}" -maxdepth 1 -type f -name "*.py" ! -name "__init__.py" -printf "%f\n" \
            | sort \
            | sed "s/\.py$//"
    )
fi

if [[ -n "${KERNEL_FILTER}" ]]; then
    mapfile -t kernels < <(printf "%s\n" "${kernels[@]}" | grep -E "${KERNEL_FILTER}" || true)
fi

if [[ "${#kernels[@]}" -eq 0 ]]; then
    echo "No kernels found to test."
    exit 1
fi

echo "=========================================="
echo "Testing Q2FP8 Attention Kernels"
echo "=========================================="
echo "Kernel directory: ${KERNEL_DIR}"
if [[ -n "${KERNELS_OVERRIDE}" ]]; then
    echo "KERNELS override: ${KERNELS_OVERRIDE}"
fi
if [[ -n "${KERNEL_FILTER}" ]]; then
    echo "KERNEL_FILTER: ${KERNEL_FILTER}"
fi
echo "Kernel count: ${#kernels[@]}"
echo "Additional arguments: $*"
echo ""

failures=()
total="${#kernels[@]}"
index=1

for kernel in "${kernels[@]}"; do
    echo "[${index}/${total}] Testing: ${kernel}"
    echo "----------------------------------------"
    if ! python "${BENCH_SCRIPT}" --attn-kernel "${kernel}" "$@"; then
        failures+=("${kernel}")
    fi
    echo ""
    index=$((index + 1))
done

echo "=========================================="
if [[ "${#failures[@]}" -eq 0 ]]; then
    echo "All kernel tests completed!"
else
    echo "Completed with failures."
    echo "Failed kernels: ${failures[*]}"
    exit 1
fi
echo "=========================================="

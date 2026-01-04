#!/bin/bash
# H100优化系统测试脚本
# 测试真正有效的优化方向：Block Size、Delta、Precomputed Threshold

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/h100_optimization_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "H100 Optimization Systematic Tests"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Log directory: ${LOG_DIR}"
echo ""

# 检查GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

if [[ ! "$GPU_NAME" =~ "H100" ]]; then
    echo "WARNING: This script is optimized for H100. Current GPU: $GPU_NAME"
    echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
    read
fi

echo ""
echo "=========================================="
echo "Test Plan"
echo "=========================================="
echo "1. Block Size Optimization (T=65536)"
echo "2. Delta Parameter Optimization (T=65536)"
echo "3. Precomputed Threshold Effect (T=65536)"
echo "4. Combined Optimizations (T=65536)"
echo "5. Scaling Test (Multiple sequence lengths)"
echo ""
echo "Estimated time: ~10-15 minutes"
echo ""
read -p "Press Enter to start tests..."

# ============================================================================
# Test 1: Block Size Optimization
# ============================================================================
echo ""
echo "=========================================="
echo "Test 1: Block Size Optimization"
echo "=========================================="
LOG_FILE="${LOG_DIR}/test1_block_size_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/test1_block_size_${TIMESTAMP}.json"

python test_h100_optimizations.py \
    --test bs \
    --T 65536 \
    --warmup 20 \
    --iters 100 \
    --output "${JSON_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "✓ Test 1 completed. Log: ${LOG_FILE}"

# ============================================================================
# Test 2: Delta Parameter Optimization
# ============================================================================
echo ""
echo "=========================================="
echo "Test 2: Delta Parameter Optimization"
echo "=========================================="
LOG_FILE="${LOG_DIR}/test2_delta_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/test2_delta_${TIMESTAMP}.json"

python test_h100_optimizations.py \
    --test delta \
    --T 65536 \
    --warmup 20 \
    --iters 100 \
    --output "${JSON_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "✓ Test 2 completed. Log: ${LOG_FILE}"

# ============================================================================
# Test 3: Precomputed Threshold Effect
# ============================================================================
echo ""
echo "=========================================="
echo "Test 3: Precomputed Threshold Effect"
echo "=========================================="
LOG_FILE="${LOG_DIR}/test3_threshold_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/test3_threshold_${TIMESTAMP}.json"

python test_h100_optimizations.py \
    --test threshold \
    --T 65536 \
    --warmup 20 \
    --iters 100 \
    --output "${JSON_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "✓ Test 3 completed. Log: ${LOG_FILE}"

# ============================================================================
# Test 4: Combined Optimizations
# ============================================================================
echo ""
echo "=========================================="
echo "Test 4: Combined Optimizations"
echo "=========================================="
LOG_FILE="${LOG_DIR}/test4_combined_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/test4_combined_${TIMESTAMP}.json"

python test_h100_optimizations.py \
    --test combined \
    --T 65536 \
    --warmup 20 \
    --iters 100 \
    --output "${JSON_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "✓ Test 4 completed. Log: ${LOG_FILE}"

# ============================================================================
# Test 5: Scaling Test (Multiple Sequence Lengths)
# ============================================================================
echo ""
echo "=========================================="
echo "Test 5: Scaling Test"
echo "=========================================="

for T in 16384 32768 65536 131072 262144; do
    echo ""
    echo "--- Testing T=${T} ---"
    LOG_FILE="${LOG_DIR}/test5_scaling_T${T}_${TIMESTAMP}.log"
    JSON_FILE="${LOG_DIR}/test5_scaling_T${T}_${TIMESTAMP}.json"

    python test_h100_optimizations.py \
        --test combined \
        --T ${T} \
        --warmup 20 \
        --iters 100 \
        --output "${JSON_FILE}" \
        2>&1 | tee "${LOG_FILE}"

    echo "✓ T=${T} completed"
done

echo "✓ Test 5 completed"

# ============================================================================
# Generate Summary Report
# ============================================================================
echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

SUMMARY_FILE="${LOG_DIR}/SUMMARY_${TIMESTAMP}.log"

{
    echo "=========================================="
    echo "H100 Optimization Test Summary"
    echo "=========================================="
    echo "Timestamp: ${TIMESTAMP}"
    echo "GPU: ${GPU_NAME}"
    echo ""

    echo "=========================================="
    echo "Test 1: Block Size Optimization Results"
    echo "=========================================="
    grep -A 3 "Speedup vs FlashAttn" "${LOG_DIR}/test1_block_size_${TIMESTAMP}.log" || echo "No results found"
    echo ""

    echo "=========================================="
    echo "Test 2: Delta Parameter Results"
    echo "=========================================="
    grep -A 3 "Speedup vs FlashAttn" "${LOG_DIR}/test2_delta_${TIMESTAMP}.log" || echo "No results found"
    echo ""

    echo "=========================================="
    echo "Test 4: Combined Optimization Results"
    echo "=========================================="
    grep -A 4 "Speedup vs" "${LOG_DIR}/test4_combined_${TIMESTAMP}.log" || echo "No results found"
    echo ""

    echo "=========================================="
    echo "Test 5: Scaling Results Summary"
    echo "=========================================="
    for T in 16384 32768 65536 131072 262144; do
        if [ -f "${LOG_DIR}/test5_scaling_T${T}_${TIMESTAMP}.log" ]; then
            echo "--- T=${T} ---"
            grep -A 2 "Speedup vs FlashAttn" "${LOG_DIR}/test5_scaling_T${T}_${TIMESTAMP}.log" | head -10
            echo ""
        fi
    done

    echo "=========================================="
    echo "Files Generated"
    echo "=========================================="
    ls -lh "${LOG_DIR}/"*${TIMESTAMP}*
    echo ""

} | tee "${SUMMARY_FILE}"

echo "✓ Summary report generated: ${SUMMARY_FILE}"

# ============================================================================
# Create Archive
# ============================================================================
echo ""
echo "=========================================="
echo "Creating Archive"
echo "=========================================="

ARCHIVE_NAME="h100_optimization_results_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${SCRIPT_DIR}/${ARCHIVE_NAME}"

cd "${SCRIPT_DIR}"
tar -czf "${ARCHIVE_NAME}" h100_optimization_logs/*${TIMESTAMP}*

echo "✓ Archive created: ${ARCHIVE_PATH}"
echo ""
echo "Archive size: $(du -h ${ARCHIVE_PATH} | cut -f1)"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED!"
echo "=========================================="
echo ""
echo "📊 Results Summary:"
echo "  - Individual logs: ${LOG_DIR}/"
echo "  - Summary report: ${SUMMARY_FILE}"
echo "  - Archive file: ${ARCHIVE_PATH}"
echo ""
echo "📤 To share results, use this file:"
echo "  ${ARCHIVE_PATH}"
echo ""
echo "📋 Quick view of summary:"
echo "  cat ${SUMMARY_FILE}"
echo ""
echo "🔍 Key findings (from Test 4):"
grep "Speedup vs FlashAttn" "${LOG_DIR}/test4_combined_${TIMESTAMP}.log" | head -5 || echo "  (Check summary file for details)"
echo ""

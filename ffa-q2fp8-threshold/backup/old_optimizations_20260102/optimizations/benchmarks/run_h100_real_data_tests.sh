#!/bin/bash
# H100真实数据测试脚本 - 使用已dump的真实LLM数据

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/h100_real_data_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "H100 Real Data Optimization Tests"
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

# 数据目录
LAYER_DATA_DIR="/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data"

echo ""
echo "Layer data directory: ${LAYER_DATA_DIR}"
echo ""

# 检查数据目录是否存在
if [ ! -d "${LAYER_DATA_DIR}" ]; then
    echo "ERROR: Layer data directory not found: ${LAYER_DATA_DIR}"
    exit 1
fi

# 测试不同序列长度
for MAX_LEN in 16384 32768 65536 131072 262144; do
    echo ""
    echo "=========================================="
    echo "Testing with max length: ${MAX_LEN}"
    echo "=========================================="

    LOG_FILE="${LOG_DIR}/real_data_T${MAX_LEN}_${TIMESTAMP}.log"
    JSON_FILE="${LOG_DIR}/real_data_T${MAX_LEN}_${TIMESTAMP}.json"

    python test_h100_real_data.py \
        --layer-data-dir "${LAYER_DATA_DIR}" \
        --layer 0 \
        --max-length ${MAX_LEN} \
        --warmup 10 \
        --iters 50 \
        --output "${JSON_FILE}" \
        2>&1 | tee "${LOG_FILE}"

    echo "✓ T=${MAX_LEN} completed. Log: ${LOG_FILE}"
done

echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

SUMMARY_FILE="${LOG_DIR}/SUMMARY_${TIMESTAMP}.log"

{
    echo "=========================================="
    echo "H100 Real Data Test Summary"
    echo "=========================================="
    echo "Timestamp: ${TIMESTAMP}"
    echo "GPU: ${GPU_NAME}"
    echo "Data: Llama-3.2-3B longbench_gov_report"
    echo ""

    for MAX_LEN in 16384 32768 65536 131072 262144; do
        LOG_FILE="${LOG_DIR}/real_data_T${MAX_LEN}_${TIMESTAMP}.log"
        if [ -f "${LOG_FILE}" ]; then
            echo ""
            echo "=========================================="
            echo "Sequence Length: ${MAX_LEN}"
            echo "=========================================="
            grep -A 5 "Skip ratio:" "${LOG_FILE}" | head -40 || echo "No results found"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Files Generated"
    echo "=========================================="
    ls -lh "${LOG_DIR}/"*${TIMESTAMP}*
    echo ""

} | tee "${SUMMARY_FILE}"

echo "✓ Summary report generated: ${SUMMARY_FILE}"

# 创建压缩包
ARCHIVE_NAME="h100_real_data_results_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${SCRIPT_DIR}/${ARCHIVE_NAME}"

cd "${SCRIPT_DIR}"
tar -czf "${ARCHIVE_NAME}" h100_real_data_logs/*${TIMESTAMP}*

echo ""
echo "✓ Archive created: ${ARCHIVE_PATH}"
echo ""
echo "Archive size: $(du -h ${ARCHIVE_PATH} | cut -f1)"
echo ""

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
echo "📤 To share results:"
echo "  ${ARCHIVE_PATH}"
echo ""
echo "📋 Quick view:"
echo "  cat ${SUMMARY_FILE}"
echo ""

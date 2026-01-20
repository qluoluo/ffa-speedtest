#!/bin/bash
# 对比 Q2FP8-Unified 与 Baseline 的 Decode 速度

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 输出目录（统一整理到时间戳子目录）
OUTPUT_ROOT="${SCRIPT_DIR}/outputs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_FILE="${RUN_DIR}/run_decode_comparison.log"
OUTPUT_JSON="${RUN_DIR}/decode_speed_comparison.json"

# 记录所有输出到日志
exec > >(tee "$LOG_FILE") 2>&1

# 默认参数
PROMPT_TYPE="medium"
MAX_NEW_TOKENS=128
NUM_RUNS=3
DEVICE="cuda:0"
K_BITS=2
DELTA=5.0
BLOCK_SIZE=128
USE_CUDAGRAPH=""
MODEL_PATH=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --k_bits)
            K_BITS="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --use_cudagraph)
            USE_CUDAGRAPH="--use_cudagraph"
            shift 1
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Running Decode Speed Comparison"
echo "========================================================================"
echo "Prompt type: $PROMPT_TYPE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Number of runs: $NUM_RUNS"
echo "Device: $DEVICE"
echo "K bits: $K_BITS"
echo "Delta: $DELTA"
echo "Block size: $BLOCK_SIZE"
echo "Use CUDA Graph: ${USE_CUDAGRAPH:-false}"
echo "Model path: ${MODEL_PATH:-<default in compare_decode_speed.py>}"
echo "Output dir: $RUN_DIR"
echo "Output JSON: $OUTPUT_JSON"
echo "Log file: $LOG_FILE"
echo "========================================================================"

MODEL_PATH_ARG=()
if [[ -n "$MODEL_PATH" ]]; then
    MODEL_PATH_ARG=(--model_path "$MODEL_PATH")
fi

python3 compare_decode_speed.py \
    --prompt_type "$PROMPT_TYPE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_runs "$NUM_RUNS" \
    --device "$DEVICE" \
    --k_bits "$K_BITS" \
    --delta "$DELTA" \
    --block_size "$BLOCK_SIZE" \
    --output "$OUTPUT_JSON" \
    "${MODEL_PATH_ARG[@]}" \
    $USE_CUDAGRAPH

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "========================================================================"

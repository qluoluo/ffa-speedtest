#!/bin/bash
# 对比 Q2FP8-Unified 与 Baseline 的 Decode 速度

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
PROMPT_TYPE="medium"
MAX_NEW_TOKENS=128
NUM_RUNS=3
DEVICE="cuda:0"
K_BITS=2
DELTA=5.0
BLOCK_SIZE=128
USE_CUDAGRAPH=""

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
echo "========================================================================"

python3 compare_decode_speed.py \
    --prompt_type "$PROMPT_TYPE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_runs "$NUM_RUNS" \
    --device "$DEVICE" \
    --k_bits "$K_BITS" \
    --delta "$DELTA" \
    --block_size "$BLOCK_SIZE" \
    $USE_CUDAGRAPH

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "========================================================================"

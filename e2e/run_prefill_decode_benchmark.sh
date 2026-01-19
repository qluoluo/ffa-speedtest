#!/bin/bash
# 运行 prefill 和 decode 阶段的详细性能测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
PROMPT_LENGTHS="32768"
DECODE_LENGTHS="256 512"
NUM_RUNS=3
DEVICE="cuda:0"
OUTPUT="prefill_decode_benchmark.json"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt_lengths)
            PROMPT_LENGTHS="$2"
            shift 2
            ;;
        --decode_lengths)
            DECODE_LENGTHS="$2"
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
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --skip_baseline)
            SKIP_BASELINE="--skip_baseline"
            shift 1
            ;;
        --skip_q2fp8)
            SKIP_Q2FP8="--skip_q2fp8"
            shift 1
            ;;
        --quick)
            # 快速测试模式
            PROMPT_LENGTHS="512 2048"
            DECODE_LENGTHS="1 32"
            NUM_RUNS=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Prefill + Decode Benchmark"
echo "========================================================================"
echo "Prompt lengths: $PROMPT_LENGTHS"
echo "Decode lengths: $DECODE_LENGTHS"
echo "Number of runs: $NUM_RUNS"
echo "Device: $DEVICE"
echo "Output: $OUTPUT"
echo "========================================================================"

# 运行测试
python3 benchmark_prefill_decode.py \
    --prompt_lengths $PROMPT_LENGTHS \
    --decode_lengths $DECODE_LENGTHS \
    --num_runs $NUM_RUNS \
    --device $DEVICE \
    --output $OUTPUT \
    $SKIP_BASELINE \
    $SKIP_Q2FP8

# 生成可视化
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "========================================================================"
    echo "Generating visualizations..."
    echo "========================================================================"
    python3 visualize_prefill_decode.py --input $OUTPUT --output_dir .
fi

echo ""
echo "========================================================================"
echo "Benchmark complete!"
echo "========================================================================"
echo "Results saved to: $OUTPUT"
echo "Plots saved to: prefill_decode_analysis.png"
echo "========================================================================"

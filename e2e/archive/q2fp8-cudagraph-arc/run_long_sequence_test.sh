#!/bin/bash
# 快速测试脚本: 16K 和 32K 长序列

set -e

# 配置
MODEL_PATH="/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"  # 需要修改为实际模型路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_long_seq"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

echo "=================================="
echo "长序列性能测试"
echo "=================================="
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "=================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行测试
cd "$SCRIPT_DIR"

python test_long_sequence.py \
    --model_path "$MODEL_PATH" \
    --prompt_lengths "16384,32768" \
    --decode_length 128 \
    --num_runs 3 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=================================="
echo "测试完成!"
echo "结果保存在: $OUTPUT_DIR"
echo "=================================="

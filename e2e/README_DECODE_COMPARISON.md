# E2E Decode Speed Comparison

对比 Q2FP8-Unified 与 Baseline (Flash Attention 2) 的 decode 速度。

## 快速开始

### 运行对比测试

```bash
# 使用默认参数（medium prompt, 128 tokens）
./run_decode_comparison.sh

# 启用 CUDA Graph 加速
./run_decode_comparison.sh --use_cudagraph

# 指定参数
./run_decode_comparison.sh \
    --prompt_type long \
    --max_new_tokens 256 \
    --num_runs 5 \
    --device cuda:0 \
    --use_cudagraph
```

### 或直接使用 Python 脚本

```bash
# 不启用 CUDA Graph
python3 compare_decode_speed.py \
    --prompt_type medium \
    --max_new_tokens 128 \
    --num_runs 3 \
    --device cuda:0

# 启用 CUDA Graph
python3 compare_decode_speed.py \
    --prompt_type medium \
    --max_new_tokens 128 \
    --num_runs 3 \
    --device cuda:0 \
    --use_cudagraph
```

## 测试参数

- `--prompt_type`: 提示类型 (short, medium, long, custom)
- `--prompt_tokens`: 自定义提示长度（仅当 prompt_type=custom 时）
- `--max_new_tokens`: 生成的 token 数量（默认 128）
- `--num_runs`: 测试运行次数（默认 3）
- `--device`: 设备（默认 cuda:0）
- `--k_bits`: 量化位数（2 或 4，默认 2）
- `--delta`: Delta 阈值（默认 5.0）
- `--block_size`: 块大小（默认 128）
- `--use_cudagraph`: 启用 CUDA Graph 加速（推荐）

## 测试不同上下文长度

```bash
# 短上下文 (~376 tokens)
python3 compare_decode_speed.py --prompt_type medium

# 长上下文 (~957 tokens)
python3 compare_decode_speed.py --prompt_type long

# 自定义上下文长度
python3 compare_decode_speed.py --prompt_type custom --prompt_tokens 2000
python3 compare_decode_speed.py --prompt_type custom --prompt_tokens 4000
python3 compare_decode_speed.py --prompt_type custom --prompt_tokens 8000
```

## 查看结果

### 1. 查看测试报告

```bash
cat DECODE_SPEED_COMPARISON_REPORT.md
```

### 2. 查看汇总结果

```bash
python3 summarize_results.py
```

### 3. 生成可视化图表

```bash
python3 visualize_results.py
# 生成 decode_speed_comparison.png
```

## 测试结果总结

### 最新结果（启用 CUDA Graph）

基于 Llama-3.1-8B 模型的测试结果：

| Context Length | Baseline (tok/s) | Q2FP8+CUDAGraph (tok/s) | Speedup | Slowdown |
|----------------|------------------|-------------------------|---------|----------|
| 376 tokens     | 33.76            | 25.83                   | 0.767x  | 1.30x    |
| 957 tokens     | 33.88            | 26.21                   | 0.774x  | 1.29x    |
| 4116 tokens    | 34.04            | 25.55                   | 0.751x  | 1.33x    |

### CUDA Graph 性能提升

对比启用/不启用 CUDA Graph 的性能差异：

| Context Length | No CUDAGraph (tok/s) | With CUDAGraph (tok/s) | Improvement |
|----------------|----------------------|------------------------|-------------|
| 376 tokens     | 22.03                | 25.83                  | **+17.2%**  |
| 957 tokens     | 20.70                | 26.21                  | **+26.6%**  |
| 4116 tokens    | 18.19                | 25.55                  | **+40.5%**  |

**关键发现：**
- CUDA Graph 为 Q2FP8-Unified 带来 **17-40% 的性能提升**
- 性能提升随上下文长度增加而增大
- 但即使启用 CUDA Graph，Q2FP8-Unified 仍比 Baseline **慢 1.30x**
- Baseline 保持稳定的 ~34 tok/s
- Q2FP8+CUDAGraph 保持 ~26 tok/s

### 之前的结果（未启用 CUDA Graph）

| Context Length | Baseline (tok/s) | Q2FP8-Unified (tok/s) | Speedup | Slowdown |
|----------------|------------------|----------------------|---------|----------|
| 376 tokens     | 33.56            | 22.09                | 0.658x  | 1.52x    |
| 957 tokens     | 34.15            | 20.70                | 0.606x  | 1.65x    |
| 2246 tokens    | 33.91            | 22.17                | 0.654x  | 1.53x    |
| 4116 tokens    | 33.68            | 18.19                | 0.540x  | 1.85x    |

## 输出文件

- `decode_speed_comparison.json`: 最新一次测试的详细结果
- `decode_speed_summary.json`: 所有测试的汇总结果
- `decode_speed_comparison.png`: 可视化对比图表
- `DECODE_SPEED_COMPARISON_REPORT.md`: 详细分析报告（无 CUDA Graph）
- `CUDAGRAPH_COMPARISON_REPORT.md`: CUDA Graph 性能分析报告（推荐阅读）

## 脚本说明

- `compare_decode_speed.py`: 主对比脚本，运行 baseline 和 Q2FP8-Unified 测试
- `run_decode_comparison.sh`: 便捷的 shell 包装脚本
- `summarize_results.py`: 汇总和展示所有测试结果
- `visualize_results.py`: 生成可视化图表

## 注意事项

1. **公平对比**：脚本使用 forward() 逐 token 生成，确保两个模型生成相同数量的 tokens
2. **不停止于 EOS**：为了公平对比，即使遇到 EOS token 也会继续生成到指定数量
3. **详细计时**：分别测量 prefill 和 decode 阶段的时间
4. **多次运行**：默认运行 3 次取平均值，减少测量误差

## 下一步

如果想要改善 Q2FP8-Unified 的性能，可以尝试：

1. **启用 CUDA Graph**（强烈推荐）
   - 使用 `--use_cudagraph` 参数
   - 可以带来 17-40% 的性能提升
   - 长上下文下效果更明显

2. 测试更长的上下文（8K-32K tokens）
   - 量化收益可能在极长上下文下才显现

3. 优化 kernel 实现
   - Profile kernel 找出性能瓶颈
   - 优化内存访问模式
   - 减少原子操作

4. 尝试不同的量化参数
   - k_bits=4（不那么激进的量化）
   - 不同的 delta 值
   - 不同的 block_size (64, 256)

5. 测试更大的 batch size
   - 量化开销可能在更大 batch 下摊销得更好

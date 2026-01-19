# Prefill + Decode 性能测试

详细测量 Q2FP8-Unified 和 Baseline 在不同输入/输出长度下的 prefill 和 decode 阶段性能。

## 功能特点

- **分离测量**: 独立测量 prefill 和 decode 阶段的耗时
- **多配置测试**: 测试多种 prompt 长度和 decode 长度的组合
- **逐 token 计时**: 记录每个 decode token 的生成时间
- **详细对比**: 生成全面的性能对比报告和可视化图表

## 快速开始

### 1. 快速测试（推荐先运行）

```bash
./run_prefill_decode_benchmark.sh --quick
```

这会测试：
- Prompt lengths: 512, 2048
- Decode lengths: 1, 32
- Runs: 1 次

### 2. 完整测试

```bash
./run_prefill_decode_benchmark.sh
```

默认配置：
- Prompt lengths: 512, 2048, 8192, 32768
- Decode lengths: 1, 32, 128, 512
- Runs: 3 次

### 3. 自定义测试

```bash
./run_prefill_decode_benchmark.sh \
    --prompt_lengths "1024 4096 16384" \
    --decode_lengths "64 256" \
    --num_runs 5 \
    --device cuda:0
```

## 参数说明

### Shell 脚本参数

- `--prompt_lengths "512 2048 8192"`: 要测试的 prompt 长度列表
- `--decode_lengths "1 32 128"`: 要测试的 decode 长度列表
- `--num_runs 3`: 每个配置运行的次数
- `--device cuda:0`: 使用的设备
- `--output filename.json`: 输出文件名
- `--skip_baseline`: 跳过 baseline 测试
- `--skip_q2fp8`: 跳过 Q2FP8 测试
- `--quick`: 快速测试模式（少量配置，1次运行）

### Python 脚本参数

直接使用 Python 脚本：

```bash
python3 benchmark_prefill_decode.py \
    --prompt_lengths 512 2048 8192 \
    --decode_lengths 1 32 128 \
    --num_runs 3 \
    --device cuda:0 \
    --output results.json
```

## 输出文件

### 1. JSON 结果文件

`prefill_decode_benchmark.json` 包含所有测试的详细数据：

```json
[
  {
    "config": {
      "prompt_length": 512,
      "decode_length": 32,
      "num_runs": 3
    },
    "baseline": {
      "prompt_length": 512,
      "num_decode_tokens": 32,
      "avg_prefill_ms": 123.45,
      "avg_decode_ms": 456.78,
      "avg_per_token_ms": 14.27,
      "memory_mb": 15234.56,
      "prefill_times": [120.1, 125.2, 125.0],
      "decode_times": [450.0, 460.0, 460.3],
      "per_token_times": [[...], [...], [...]]
    },
    "q2fp8": {
      ...
    }
  },
  ...
]
```

### 2. 可视化图表

`prefill_decode_analysis.png` 包含 9 个子图：

1. **Prefill Time vs Prompt Length**: 不同 decode 长度下的 prefill 时间
2. **Decode Time vs Decode Length**: 不同 prompt 长度下的 decode 时间
3. **Prefill Speedup Ratio**: Prefill 阶段的加速比
4. **Decode Speedup Ratio**: Decode 阶段的加速比
5. **Per-Token Decode Time**: 每个 token 的平均生成时间
6. **Memory Usage**: 内存使用对比
7. **Decode Speedup Heatmap**: Decode 加速比热力图
8. **Prefill Speedup Heatmap**: Prefill 加速比热力图
9. **Time Breakdown**: 时间分解（prefill vs decode）

### 3. 终端输出

运行时会在终端输出详细的对比表格：

```
====================================================================================================
RESULTS: Prompt Length = 512, Decode Length = 32
====================================================================================================

--- PREFILL PHASE ---
Metric                         Baseline             Q2FP8                Ratio
-------------------------------------------------------------------------------------
Prefill Time (ms)              123.45               156.78               0.787x
Prefill Throughput (tok/s)     4147.23              3265.12              0.787x

--- DECODE PHASE ---
Metric                         Baseline             Q2FP8                Ratio
-------------------------------------------------------------------------------------
Total Decode Time (ms)         456.78               589.23               0.775x
Per-Token Time (ms)            14.27                18.41                0.775x
Decode Throughput (tok/s)      70.07                54.32                0.775x

--- TOTAL (PREFILL + DECODE) ---
Metric                         Baseline             Q2FP8                Ratio
-------------------------------------------------------------------------------------
Total Time (ms)                580.23               746.01               0.778x
Memory (MB)                    15234.56             15678.90             1.029x

====================================================================================================
SUMMARY: Q2FP8 is 1.29x SLOWER in decode (54.32 vs 70.07 tok/s)
====================================================================================================
```

## 使用场景

### 场景 1: 诊断性能瓶颈

找出 Q2FP8 在哪个阶段慢：

```bash
./run_prefill_decode_benchmark.sh --prompt_lengths "4096 8192 16384" --decode_lengths "128"
```

查看结果：
- 如果 prefill 慢很多 → 量化开销太大
- 如果 decode 慢很多 → kernel 性能问题
- 如果两者都慢 → 整体实现问题

### 场景 2: 找到最佳工作点

测试多种配置找出 Q2FP8 有优势的场景：

```bash
./run_prefill_decode_benchmark.sh \
    --prompt_lengths "1024 2048 4096 8192 16384 32768 65536" \
    --decode_lengths "1 16 32 64 128 256 512"
```

查看热力图找出加速比 > 1.0 的区域。

### 场景 3: 对比不同 decode 长度

固定 prompt，测试不同 decode 长度：

```bash
./run_prefill_decode_benchmark.sh \
    --prompt_lengths "8192" \
    --decode_lengths "1 10 50 100 200 500 1000"
```

### 场景 4: 只测试 baseline 或 Q2FP8

```bash
# 只测试 baseline
./run_prefill_decode_benchmark.sh --skip_q2fp8

# 只测试 Q2FP8
./run_prefill_decode_benchmark.sh --skip_baseline
```

## 可视化结果

生成可视化后，可以单独重新生成图表：

```bash
python3 visualize_prefill_decode.py \
    --input prefill_decode_benchmark.json \
    --output_dir ./plots
```

## 注意事项

1. **内存要求**: 长 prompt (32K+) 需要大量 GPU 内存
2. **测试时间**: 完整测试可能需要 1-2 小时
3. **建议顺序**: 先运行 `--quick` 测试，确认正常后再运行完整测试
4. **结果缓存**: 每次运行会覆盖之前的结果文件

## 示例：诊断 32K prompt 性能问题

```bash
# 1. 测试 32K prompt 的 prefill 和 decode
./run_prefill_decode_benchmark.sh \
    --prompt_lengths "32768" \
    --decode_lengths "1 32 128" \
    --num_runs 3

# 2. 查看结果
cat prefill_decode_benchmark.json | python3 -m json.tool

# 3. 查看可视化
# 打开 prefill_decode_analysis.png
```

预期发现：
- Prefill 阶段 Q2FP8 慢 ~40%（量化开销）
- Decode 阶段 Q2FP8 慢 ~3x（kernel 性能问题）

## 与 Kernel Benchmark 对比

这个 E2E 测试与 kernel benchmark 的区别：

| 特性 | Kernel Benchmark | E2E Benchmark |
|------|------------------|---------------|
| 测试范围 | 单个 attention layer | 完整模型 (32 layers + MLP) |
| 包含内容 | 纯 attention kernel | Prefill + Decode + 量化 + 所有 layers |
| 速度 | 快 (ms 级别) | 慢 (秒级别) |
| 结果 | Kernel 性能 | 实际应用性能 |

**关键差异**: Kernel benchmark 显示 Q2FP8 快 4x，但 E2E 显示慢 3x，说明问题在 kernel 之外（量化、cache 管理等）。

## 故障排查

### 问题 1: CUDA out of memory

减少 prompt 长度或 decode 长度：

```bash
./run_prefill_decode_benchmark.sh \
    --prompt_lengths "512 2048" \
    --decode_lengths "32"
```

### 问题 2: 测试太慢

使用快速模式或减少运行次数：

```bash
./run_prefill_decode_benchmark.sh --quick
# 或
./run_prefill_decode_benchmark.sh --num_runs 1
```

### 问题 3: 导入错误

确保路径正确：

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/e2e
./run_prefill_decode_benchmark.sh --quick
```

## 相关文档

- `README_DECODE_COMPARISON.md`: 原始的 decode 速度对比
- `CUDAGRAPH_COMPARISON_REPORT.md`: CUDA Graph 性能分析
- `SUMMARY.md`: 完整的性能分析总结

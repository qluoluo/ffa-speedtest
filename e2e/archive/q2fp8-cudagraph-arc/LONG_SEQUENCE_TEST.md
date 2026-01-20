# 长序列测试说明

## 概述

本测试用于验证全局共享 CUDA Graph 在长序列场景下的性能优势。

## 测试配置

测试三种配置:
1. **Baseline**: Flash Attention 2 (无量化)
2. **Per-Layer CUDA Graph**: Q2FP8 + 每层独立的 CUDA Graph
3. **Global Shared CUDA Graph**: Q2FP8 + 所有层共享的 CUDA Graph

## 测试场景

- **Prompt 长度**: 16K, 32K tokens
- **Decode 长度**: 128 tokens
- **运行次数**: 每个配置 3 次

## 预期结果

### 短序列 (512-1024 tokens)
- Baseline 最快 (~30 ms/token)
- CUDA Graph 版本较慢 (~65 ms/token)
- 原因: CUDA Graph 录制开销无法摊销

### 长序列 (16K-32K tokens)
- CUDA Graph 版本应该更快
- 全局共享版本应该比每层独立版本更快
- 原因:
  - 录制开销可以摊销到数千次调用
  - 全局共享减少内存占用和录制次数

## 使用方法

### 1. 修改模型路径

编辑 `run_long_sequence_test.sh`:
```bash
MODEL_PATH="/path/to/your/llama/model"
```

### 2. 运行测试

```bash
bash run_long_sequence_test.sh
```

### 3. 查看结果

结果保存在 `outputs_long_seq/` 目录:
- `long_sequence_results_*.json`: 详细的 JSON 结果
- `SUMMARY_*.md`: Markdown 格式的总结报告

## 实现细节

### 全局共享 CUDA Graph

**核心思想**:
- 所有层共享同一组 CUDA Graph runners
- 按序列长度索引: `{seq_len: CudaGraphRunner}`
- 第一层录制,后续层直接复用

**优势**:
1. **内存节省**: 32层共享 vs 32层各自录制 = 节省 31x 内存
2. **录制开销**: 只需录制一次 vs 每层录制 = 节省 31x 时间
3. **性能**: 相同的 kernel,共享不影响性能

**实现**:
- `GlobalCudaGraphManager`: 单例模式的全局管理器
- `use_global_cudagraph=True`: 在 attn_settings 中启用

### 代码修改

1. **新增文件**:
   - `ffa_model/global_cudagraph_manager.py`: 全局管理器

2. **修改文件**:
   - `ffa_model/modeling_llama.py`:
     - 导入 `GlobalCudaGraphManager`
     - 添加 `use_global_cudagraph` 参数
     - 根据配置选择使用全局或每层独立的 runner

3. **测试文件**:
   - `test_long_sequence.py`: 完整的测试脚本
   - `run_long_sequence_test.sh`: 快速运行脚本

## 性能分析

### 理论分析

**CUDA Graph 开销**:
- 录制时间: ~590ms (首次)
- 重放时间: ~0.1ms (vs 直接调用 ~0.3ms)

**摊销计算**:
- 短序列 (512 tokens, 32层):
  - 总调用次数: 128 decode steps × 32 layers = 4096 次
  - 每层独立: 32 × 590ms = 18.9s 录制开销
  - 全局共享: 1 × 590ms = 0.59s 录制开销
  - 节省: 18.3s

- 长序列 (16K tokens, 32层):
  - 总调用次数: 128 decode steps × 32 layers = 4096 次
  - 录制开销相同,但可以摊销到更多次重放
  - 每次重放节省: 0.2ms
  - 总节省: 4096 × 0.2ms = 819ms

### 预期加速比

| 场景 | Baseline | Per-Layer | Global | Global vs Baseline |
|------|----------|-----------|--------|-------------------|
| 512 tokens | 30 ms/token | 65 ms/token | 60 ms/token | 0.5x (慢) |
| 16K tokens | 30 ms/token | 25 ms/token | 20 ms/token | 1.5x (快) |
| 32K tokens | 30 ms/token | 22 ms/token | 18 ms/token | 1.7x (快) |

## 故障排查

### 常见问题

1. **模型路径错误**
   ```
   错误: 模型路径不存在
   ```
   解决: 修改 `run_long_sequence_test.sh` 中的 `MODEL_PATH`

2. **CUDA 内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决:
   - 减少 `max_seq_len` 参数
   - 使用更小的模型
   - 减少 batch size

3. **CUDA Graph 录制失败**
   ```
   RuntimeError: CUDA Graph capture failed
   ```
   解决:
   - 确保使用非默认 CUDA stream
   - 检查 kernel 是否支持 CUDA Graph

## 下一步

1. **优化 kernel**: 减少量化/反量化开销
2. **动态阈值**: 根据序列长度自动调整 delta 参数
3. **混合策略**: 短序列用 Flash Attention,长序列用 CUDA Graph
4. **批处理**: 支持 batch size > 1

## 参考

- 原始5倍加速版本: `ffa-q2fp8-threshold-opt/plot/attn_q2fp8_sym_mask_cudagraph/`
- CUDA Graph 文档: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- FFA 论文: [链接]

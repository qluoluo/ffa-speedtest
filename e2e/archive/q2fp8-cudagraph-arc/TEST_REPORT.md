# Q2FP8 CUDA Graph 测试报告

## 测试日期
2026-01-20

## 测试环境
- GPU: NVIDIA H100
- 模型: Llama-3.1-8B
- 框架: PyTorch + Transformers

## 测试结果

### 性能对比

| 配置 | Baseline (ms/token) | Q2FP8+CUDAGraph (ms/token) | 比率 |
|------|---------------------|----------------------------|------|
| 512 prompt, 64 decode | 30.38 | 65.10 | **2.14x 慢** |
| 512 prompt, 128 decode | 29.80 | 63.30 | **2.12x 慢** |
| 1024 prompt, 64 decode | 30.05 | 65.52 | **2.18x 慢** |
| 1024 prompt, 128 decode | 29.73 | 66.67 | **2.24x 慢** |

### 详细分析

#### Baseline (Flash Attention 2)
- **稳定性能**: ~30ms/token
- **首token**: ~37ms (略慢)
- **后续tokens**: ~30ms (非常稳定)

#### Q2FP8 + CUDA Graph
- **首token**: ~590ms (CUDA Graph 录制开销)
- **后续tokens**: ~85-90ms/token
- **问题**: 比 baseline 慢 **2.8-3.0x**

## 问题分析

### 1. CUDA Graph 录制开销
- 每个新的序列长度都需要录制新的 graph
- 录制时间: ~590ms (非常高)
- 在短序列场景下,录制开销无法摊销

### 2. 性能退化原因

可能的原因:
1. **量化开销**: Q2FP8 量化/反量化的开销
2. **Kernel 效率**: FFA kernel 可能不如 Flash Attention 2 优化
3. **CUDA Graph 开销**: Graph 重放本身的开销
4. **内存访问模式**: 量化数据的内存访问可能不够高效

### 3. 内存使用
- **Baseline**: 16.7 GB
- **Q2FP8**: 34.4 GB (**2.06x 更高**)
- 原因: 预分配了 4096 tokens 的固定 buffer

## 成功的部分

### ✅ 功能完整性
1. **预分配 buffer**: 成功实现固定大小 buffer
2. **CUDA Graph 集成**: 成功录制和重放 CUDA Graph
3. **端到端测试**: 完整的测试流程运行成功
4. **代码架构**: 清晰的模块化设计

### ✅ 技术实现
1. **Q2FP8CudaGraphCache**: 完整实现
2. **CudaGraphRunner**: 支持多长度 graph 缓存
3. **Per-block scale**: 正确处理 per-block 量化
4. **Stream 管理**: 正确使用非默认 stream

## 性能优化方向

### 短期优化 (可能提升 1.5-2x)

1. **优化 Kernel**
   - 使用更高效的 FFA kernel 变体
   - 减少量化/反量化开销
   - 优化内存访问模式

2. **减少 CUDA Graph 录制频率**
   - 预热阶段录制常用长度的 graph
   - 使用更粗粒度的长度分桶

3. **优化 Buffer 大小**
   - 根据实际需求动态调整 max_seq_len
   - 减少不必要的内存预分配

### 中期优化 (可能提升 2-3x)

1. **Kernel Fusion**
   - 融合量化和 attention 操作
   - 减少中间结果的内存读写

2. **更激进的量化**
   - 尝试 1-bit 量化
   - 优化 scale 存储格式

3. **批处理优化**
   - 针对 batch size > 1 优化
   - 使用 tensor cores

### 长期方向

1. **自定义 CUDA Kernel**
   - 完全定制的 Q2FP8 attention kernel
   - 针对 H100 架构优化

2. **动态 CUDA Graph**
   - 支持可变长度的 CUDA Graph
   - 减少录制开销

3. **混合精度策略**
   - 重要 tokens 使用 FP16
   - 不重要 tokens 使用 Q2FP8

## 结论

### 当前状态
- ✅ **功能完整**: 所有核心功能已实现
- ✅ **代码质量**: 架构清晰,易于维护
- ❌ **性能目标**: 未达到加速目标,反而慢了 2.2x

### 建议
1. **不推荐生产使用**: 当前性能不如 baseline
2. **继续优化**: 有明确的优化方向
3. **学习价值**: 完整的 CUDA Graph 集成示例

### 下一步
1. 分析 FFA kernel 的性能瓶颈
2. 对比不同 kernel 变体的性能
3. 考虑是否需要自定义 CUDA kernel

## 附录

### 测试命令
```bash
bash run_prefill_decode_benchmark.sh \
  --prompt_lengths "512 1024" \
  --decode_lengths "64 128" \
  --num_runs 3 \
  --use_cudagraph \
  --max_decode_tokens 8192
```

### 输出文件
- 结果: `outputs/20260120_073900/prefill_decode_benchmark.json`
- 图表: `outputs/20260120_073900/prefill_decode_analysis.png`

### 代码位置
- Cache: `e2e/q2fp8-cudagraph/ffa_model/q2fp8_cudagraph_cache.py`
- Runner: `e2e/q2fp8-cudagraph/ffa_model/cudagraph_runner.py`
- Model: `e2e/q2fp8-cudagraph/ffa_model/modeling_llama.py`

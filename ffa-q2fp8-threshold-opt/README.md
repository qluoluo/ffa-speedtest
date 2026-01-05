# Q2FP8 Attention Kernel H100 Optimizations

针对 H100 GPU 的 Q2FP8 量化 Attention Kernel 性能优化项目。

## 快速开始

### 运行测试

```bash
python test_optimizations.py
```

### 使用优化 Kernel

```python
from attn_kernel.attn_kernel_opt1_compact import CUDAGraphDecodeRunnerOpt1Compact

# 创建 runner
runner = CUDAGraphDecodeRunnerOpt1Compact(
    q, k_q, k_scale, k_zero, v,
    k_residual=k_residual,
    BS=128,
    delta=5.0,
    use_fp8_residual=True,
)

# 推理
output = runner.replay(q, k_q, k_scale, k_zero, v, k_residual=k_residual)
```

## 项目结构

```
.
├── H100_PERFORMANCE_ANALYSIS.md     # 详细性能分析
├── OPTIMIZATION_SUMMARY.md          # 优化实现指南
├── PROJECT_SUMMARY.md               # 项目完成总结
├── test_optimizations.py            # 测试脚本
└── attn_kernel/
    ├── README_OPTIMIZATIONS.md      # 优化说明
    ├── attn_kernel_opt1_compact.py  # ✅ Opt1: Stage2 Compact
    ├── attn_kernel_opt2_twostage.py # 🚧 Opt2: Two-Stage Selector
    ├── attn_kernel_opt3_qreuse.py   # 🚧 Opt3: Q Reuse
    ├── attn_kernel_opt4_fused.py    # 🚧 Opt4: Fused Threshold
    └── attn_kernel_opt5_async.py    # 🚧 Opt5: Async Pipeline
```

## 核心问题

当前 H100 性能瓶颈：
- ❌ Skip 99% blocks 但只有 1.4x 加速
- ❌ 额外开销占 86% (0.223ms)
- ❌ Stage2 串行遍历 2048 iterations
- ❌ Stage1 所有 blocks 都计算 selector

## 优化方案

### ✅ Opt1: Stage2 Compact Indices (已实现)

**收益**: ~0.05ms | **加速**: 1.37x → 1.8x

**核心改进**:
- 压缩 mask_buf 为 kept_indices
- Stage2 只遍历 ~20 个 kept blocks (原 2048)
- 减少 100x iterations

### 🚧 Opt2: Two-Stage Selector (下一步)

**收益**: ~0.10ms | **加速**: 1.8x → 3.5x

**核心改进**:
- Coarse filter 快速过滤 95% blocks
- Fine selector 只对 5% survivors
- Selector 计算量减少 ~10x

### 🚧 Opt3-5: 其他优化

详见 `OPTIMIZATION_SUMMARY.md`

## 性能预期 (H100 @ 262k)

| 优化 | 延迟 | vs Flash | 状态 |
|-----|------|---------|------|
| Baseline | 0.259ms | 1.37x | - |
| +Opt1 | 0.20ms | 1.8x | ✅ |
| +Opt1+2 | 0.10ms | 3.5x | 🎯 目标 |
| +Opt1+2+3 | 0.08ms | 4.4x | 未来 |
| +All | 0.05ms | 7.1x | 未来 |

## 测试结果

### RTX 4090

```
Baseline:   0.150ms @ T=10k
Opt1:       0.197ms @ T=10k (skip_ratio=0%)
```

**注意**: Skip ratio=0 因为测试配置，实际 H100 长序列会有显著 skip。

## 文档

### 核心文档

1. **[H100_PERFORMANCE_ANALYSIS.md](H100_PERFORMANCE_ANALYSIS.md)**
   - 详细性能分析
   - 瓶颈识别
   - 优化方向

2. **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)**
   - 每个优化的实现思路
   - 伪代码示例
   - 实现难度评估

3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**
   - 项目完成总结
   - 测试结果
   - 下一步计划

### Kernel 文档

4. **[attn_kernel/README_OPTIMIZATIONS.md](attn_kernel/README_OPTIMIZATIONS.md)**
   - Kernel 使用说明
   - API 文档
   - 配置参数

## 下一步

### 立即行动

1. **在 H100 上测试 Opt1**
   ```bash
   python test_optimizations.py
   ```

2. **调整参数获得更高 skip_ratio**
   - 修改 delta 值
   - 目标: skip_ratio 50-99%

### 短期计划

3. **实现 Opt2: Two-Stage Selector**
   - 预期收益: 最高 ROI
   - 目标: 3-4x vs FlashAttn

4. **组合优化测试**
   - Opt1 + Opt2
   - 端到端性能验证

## 环境要求

- CUDA >= 12.0
- PyTorch >= 2.0
- Triton >= 2.0
- GPU: H100 (推荐) 或 A100/4090 (测试)

## 开发

### 运行测试

```bash
python test_optimizations.py
```

### 添加新优化

1. 创建 `attn_kernel/attn_kernel_optN_xxx.py`
2. 实现核心 kernel 和 CUDAGraph wrapper
3. 在 `test_optimizations.py` 中添加测试
4. 更新文档

## 性能调优

### 关键参数

- `BS`: Block size (default 128)
- `SBS`: Sub-block size (default 128)
- `delta`: Threshold margin (default 5.0)
- `max_kept`: Maximum kept blocks (Opt1, default 256)

### Profiling

使用 Nsight Compute:
```bash
ncu --set full python test_optimizations.py
```

## 常见问题

### Q: Opt1 比 baseline 慢？

A: 当 skip_ratio 很低时，compact kernel 的开销超过收益。需要调整 delta 参数增加 skip_ratio。

### Q: 如何在生产环境使用？

A:
```python
# 1. 预先创建 runner
runner = CUDAGraphDecodeRunnerOpt1Compact(...)

# 2. 推理时直接 replay (zero overhead)
output = runner.replay(...)
```

### Q: H100 vs A100 性能差异？

A: H100 有更高的 memory bandwidth 和 FP8 支持，预期收益更大。

## License

内部研究项目

## 联系

项目问题请提 issue 或联系项目负责人。

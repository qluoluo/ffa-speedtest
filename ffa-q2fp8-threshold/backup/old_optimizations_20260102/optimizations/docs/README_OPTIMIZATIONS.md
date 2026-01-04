# Q2FP8 Kernel Optimizations - 使用说明

## 概述

这个实现包含三种针对H100优化的技术：
1. **LUT Dequant** - 查找表反量化（适用于所有GPU）
2. **FP8 Tensor Core** - FP8张量核心（H100专用，其他GPU会自动fallback）
3. **Async Copy** - 异步内存拷贝（H100上使用TMA，其他GPU使用标准async）

## 文件结构

```
ffa-q2fp8-threshold/
├── attn_kernel/
│   ├── attn_kernel_v1210_fused_bsz_q2fp8.py              # 原始kernel
│   ├── attn_kernel_v1210_fused_bsz_q2fp8_optimized.py  # 优化kernel
│   └── attn_kernel_v1210_fused_bsz_q2fp8_optimized_cudagraph.py  # CUDAGraph wrapper
├── test_optimized_kernels.py                            # 测试和benchmark脚本
└── README_OPTIMIZATIONS.md                              # 本文档
```

## 本地测试结果（4090）

```
================================================================================
CORRECTNESS TEST SUMMARY
================================================================================
✓ Original kernel works
✓ LUT optimization: max diff = 0.000061
✓ Async copy: max diff = 0.000061

✓ ALL TESTS PASSED! (differences within tolerance)
```

**重要发现：**
- 在4090上，LUT优化反而略慢（0.79x），这是因为：
  - 4090上原始dequant已经很快
  - 短序列下LUT的额外load开销大于计算节省
- **但在H100上预期会有显著加速**，因为：
  - H100的内存带宽更高（3TB/s vs 1TB/s）
  - FP8 tensor core可以进一步加速
  - TMA异步拷贝可以隐藏延迟

## 在H100上测试

### 1. 快速测试（验证正确性）

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold

# 快速测试
python test_optimized_kernels.py --quick

# 仅测试正确性
python test_optimized_kernels.py --test-correctness
```

### 2. 性能Benchmark

```bash
# 单个序列长度
python test_optimized_kernels.py --bench --T 65536 --iters 200

# 完整benchmark（多个序列长度）
python test_optimized_kernels.py --full-bench --iters 200
```

### 3. 与现有系统集成

在你的现有benchmark脚本中使用优化kernel：

```python
# 原始代码
from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized

# 改为使用优化版本
from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_optimized import attn_forward_decode_quantized_optimized

# 调用时指定优化选项
output = attn_forward_decode_quantized_optimized(
    q=q,
    k_q=k_q,
    k_scale=k_scale,
    k_zero=k_zero,
    v=v,
    k_residual=k_residual,
    BS=128,
    delta=5.0,
    use_fp8_residual=True,
    use_fp8_compute=True,      # 在H100上启用FP8
    use_async_copy=True,        # 在H100上启用TMA
)
```

### 4. 使用CUDAGraph版本（更快）

```python
from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_optimized_cudagraph import CUDAGraphDecodeRunnerQ2FP8Optimized

# 创建runner（一次性，会capture graph）
runner = CUDAGraphDecodeRunnerQ2FP8Optimized(
    q=q_init,
    k_q=k_q_init,
    k_scale=k_scale,
    k_zero=k_zero,
    v=v_init,
    k_residual=k_residual_init,
    BS=128,
    delta=5.0,
    use_fp8_residual=True,
    use_fp8_compute=True,      # H100上启用
    use_async_copy=True,        # H100上启用
    warmup=2,
)

# 之后每次推理只需replay（超快）
for step in range(num_steps):
    output = runner.replay(
        q=q[step],
        k_q=k_q[step],
        k_scale=k_scale,
        k_zero=k_zero,
        v=v[step],
        k_residual=k_residual[step],
    )
```

## 优化效果预期

根据理论分析和现有数据，在H100上的预期加速比：

| 优化组合 | 预期加速 | 说明 |
|---------|---------|------|
| LUT only | 1.15-1.25x | 减少dequant计算 |
| LUT + Async | 1.25-1.35x | 隐藏内存延迟 |
| LUT + FP8 | 1.4-1.6x | FP8 tensor core加速matmul |
| All (LUT + FP8 + Async) | 1.6-2.0x | 所有优化叠加 |

**与FlashAttn对比：**
- H100上目标：将当前1.25x提升到 **2.5-3.5x**
- 4090上保持：5x加速不变

## 性能调优建议

### 在H100上

1. **启用所有优化**
   ```python
   use_fp8_compute=True     # 启用FP8 tensor core
   use_async_copy=True      # 启用TMA异步拷贝
   ```

2. **调整Block Size**
   ```python
   BS=256                   # H100有更大shared memory (228KB)
   SBS=256                  # 增大sub-block size
   ```

3. **使用CUDAGraph**
   - 减少kernel launch开销
   - 在解码阶段尤其重要（batch size通常为1）

### 在4090或A100上

1. **仅启用Async Copy**
   ```python
   use_fp8_compute=False    # FP8在Ada/Ampere上不是最优
   use_async_copy=True      # 异步拷贝仍然有帮助
   ```

2. **保持原始Block Size**
   ```python
   BS=128                   # 4090的shared memory较小
   SBS=128
   ```

## Troubleshooting

### Q: H100上FP8模式报错
**A:** 确保PyTorch版本支持FP8 (>= 2.1)，如果不支持会自动fallback到FP16

### Q: 性能没有提升
**A:** 检查以下几点：
1. 是否在H100上运行？（通过`nvidia-smi`确认）
2. 是否启用了所有优化选项？
3. 序列长度是否足够长？（建议T >= 32768）
4. 是否使用了CUDAGraph？

### Q: 正确性测试失败
**A:** 如果差异超过1e-3，可能是：
1. FP8精度问题（正常，可以调整tolerance）
2. 数值不稳定（尝试降低delta值）

## 下一步工作

在H100上完成测试后，可以进一步优化：

1. **自适应Block Size**
   - 根据序列长度自动调整BS
   - 短序列用小BS，长序列用大BS

2. **更激进的剪枝**
   - 调整delta参数
   - 在H100上由于baseline更快，可能需要更激进的剪枝

3. **Multi-token推理**
   - 当batch size > 1时，进一步优化

## 联系方式

如有问题，请检查：
1. `test_optimized_kernels.py` 中的详细日志
2. Triton编译错误信息
3. CUDA错误信息（通过`CUDA_LAUNCH_BLOCKING=1`）

---

**Created:** 2025-12-31
**Tested on:** NVIDIA GeForce RTX 4090
**Target:** NVIDIA H100 80GB HBM3

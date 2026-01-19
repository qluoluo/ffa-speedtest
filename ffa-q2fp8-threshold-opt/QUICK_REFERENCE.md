# 优化工作总结 - 快速参考

## 已完成的优化

### 1️⃣ MAX_KEPT_RATIO 优化 ✅ **推荐采用**

**性能提升**: 7.3% (0.196ms → 0.182ms)

**如何使用**:
```python
# 在 attn_q2fp8_sym_lr64_atomic_compact.py 中
# 修改默认值从 0.2 改为 0.02
def attn_forward_decode_quantized(
    ...
    max_kept_ratio: float = 0.02,  # 从 0.2 改为 0.02
    ...
):
```

**或在调用时指定**:
```bash
python scripts/run_attn_bench_q2fp8_cudagraph.py \
    --attn-kernel attn_q2fp8_sym_lr64_atomic_compact \
    --max-kept-ratio 0.02
```

**详细报告**: `MAX_KEPT_OPTIMIZATION_REPORT.md`

---

### 2️⃣ Stage2 向量化优化 ⚠️ **不推荐采用**

**性能提升**: 0.8% (0.182ms → 0.180ms)

**结论**: 收益太小，不值得增加代码复杂度

**详细报告**: `STAGE2_VECTORIZATION_REPORT.md`

---

## 性能对比总结

| 版本 | 延迟 (256K) | vs Flash | vs Baseline |
|------|-------------|----------|-------------|
| Baseline (ratio=0.2) | 0.196 ms | 5.75x | - |
| + MAX_KEPT优化 (ratio=0.02) | 0.182 ms | 6.14x | **+7.3%** ✅ |
| + Stage2向量化 | 0.180 ms | 6.19x | +8.2% |

**推荐配置**: 只采用 MAX_KEPT 优化 (ratio=0.02)

---

## 下一步优化建议

### 优先级排序

1. **动态 MAX_KEPT** (最优先)
   - 根据实际 skip_ratio 自适应调整
   - 预期提升: 5-10%
   - 实现难度: 中等

2. **Warp 参数调优**
   - 测试不同的 num_warps/num_stages 组合
   - 预期提升: 5-10%
   - 实现难度: 简单

3. **优化 Stage1**
   - 优化 Q·K 计算和阈值判断
   - 预期提升: 10-15%
   - 实现难度: 较高

4. **自适应 BK 分块**
   - 根据序列长度选择最优 BK
   - 预期提升: 3-5%
   - 实现难度: 中等

---

## 快速测试命令

### 测试 MAX_KEPT 优化
```bash
# 快速测试
bash scripts/run_max_kept_quick.sh

# 完整测试
bash scripts/run_max_kept_sweep.sh

# 生成对比图
python scripts/plot_max_kept_comparison.py
```

### 测试向量化优化
```bash
# 对比原始 vs 向量化
bash scripts/compare_vectorization.sh

# 分析结果
python scripts/analyze_vectorization.py
```

---

## 文件索引

### 报告文档
- `MAX_KEPT_OPTIMIZATION_REPORT.md` - MAX_KEPT 优化完整报告
- `STAGE2_VECTORIZATION_REPORT.md` - 向量化优化实验报告
- `STAGE2_VECTORIZATION_EXPLAINED.md` - 向量化技术详解
- `README_MAX_KEPT_OPTIMIZATION.md` - 快速开始指南
- `QUICK_REFERENCE.md` - 本文件

### Kernel 文件
- `attn_kernel/attn_q2fp8_sym_lr64_atomic_compact.py` - 原始 kernel
- `attn_kernel/attn_q2fp8_sym_lr64_atomic_compact_vec.py` - 向量化 kernel (不推荐)

### 测试脚本
- `scripts/run_max_kept_sweep.sh` - MAX_KEPT 完整测试
- `scripts/run_max_kept_quick.sh` - MAX_KEPT 快速测试
- `scripts/compare_vectorization.sh` - 向量化对比测试
- `scripts/plot_max_kept_comparison.py` - 生成对比图表
- `scripts/analyze_vectorization.py` - 分析向量化结果

### 测试数据
- `plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/` - 原始 kernel 结果
- `plot/attn_q2fp8_sym_lr64_atomic_compact_vec_cudagraph/` - 向量化 kernel 结果

---

## 关键发现

### ✅ 有效的优化
- **减小 MAX_KEPT**: 从 410 → 41 (90% 内存减少)
- **减少循环次数**: Stage2 循环从 410 → 41 次
- **更好的 cache locality**: 更小的 buffer 更容易 fit in L2

### ❌ 无效的优化
- **Stage2 向量化**: 收益太小 (0.8%)
- **原因**: Stage2 不是瓶颈，已经很高效

### 💡 经验教训
1. **先优化参数，再优化代码** - MAX_KEPT 参数调优比代码优化更有效
2. **找准瓶颈** - Stage2 不是瓶颈，优化它收益有限
3. **权衡复杂度** - 0.8% 提升不值得增加代码复杂度

---

## 联系方式

如有问题，请查看详细报告或测试脚本。

生成时间: 2026-01-19
实验环境: RTX 4090, CUDA 12.x, Triton

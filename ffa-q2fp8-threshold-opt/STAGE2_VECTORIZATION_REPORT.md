# Stage2 向量化优化实验报告

## 实验目标

测试 Stage2 kernel 的向量化优化是否能带来性能提升。

## 实验配置

- **GPU**: NVIDIA GeForce RTX 4090 (48GB)
- **原始 kernel**: `attn_q2fp8_sym_lr64_atomic_compact`
- **向量化 kernel**: `attn_q2fp8_sym_lr64_atomic_compact_vec`
- **参数**: BS=128, SBS=128, delta=5.0, max_kept_ratio=0.02
- **测试**: 300 iterations, 50 warmup, step=65536

## 优化方法

### 原始实现（串行）

```python
for i in range(MAX_KEPT):  # 循环 41 次 (max_kept_ratio=0.02)
    mask_i = i < n_kept
    m_b = tl.load(compact_m_buf + i, mask=mask_i, other=neg_inf)
    l_b = tl.load(compact_l_buf + i, mask=mask_i, other=0.0)
    o_b = tl.load(compact_o_buf + i * V + v_offs, mask=mask_i, other=0.0)
    # merge...
```

### 向量化实现（展开）

```python
VEC_SIZE = 8
num_vec_blocks = (MAX_KEPT + VEC_SIZE - 1) // VEC_SIZE  # 6 次外层循环

for vec_idx in range(num_vec_blocks):
    start_i = vec_idx * VEC_SIZE

    # 内层循环使用 tl.static_range 展开
    for local_j in tl.static_range(VEC_SIZE):  # 编译时展开
        i = start_i + local_j
        mask_i = i < n_kept
        # load and merge...
```

**关键改进：**
- 外层循环次数：41 → 6 (减少 85%)
- 内层循环使用 `tl.static_range` 编译时展开
- 更好的指令流水线和循环展开

## 实验结果

### 性能对比 (256K tokens)

| 指标 | 原始版本 | 向量化版本 | 改进 |
|------|----------|------------|------|
| **延迟** | 0.182 ms | 0.180 ms | **+0.8%** ⬆️ |
| **vs Flash** | 6.14x | 6.19x | +0.8% |

### 不同序列长度表现

| 长度 | 原始 (ms) | 向量化 (ms) | 改进 |
|------|-----------|-------------|------|
| 64K  | 0.056     | 0.056       | +0.1% |
| 128K | 0.092     | 0.092       | -0.1% |
| 192K | 0.136     | 0.136       | -0.1% |
| 256K | 0.182     | 0.180       | **+0.8%** |

## 结果分析

### 为什么提升不明显？

1. **MAX_KEPT 已经很小**
   - 使用 max_kept_ratio=0.02 后，MAX_KEPT=41
   - 原始循环只有 41 次，本身就不是主要瓶颈

2. **Stage2 占比较小**
   - Stage1 (计算 attention) 占大部分时间
   - Stage2 (merge) 只占总时间的一小部分

3. **内存访问模式**
   - 即使展开循环，内存访问模式没有本质改变
   - 仍然是串行的 merge 操作（数据依赖）

4. **Triton 限制**
   - 无法使用真正的向量化加载（会导致索引错误）
   - 只能通过循环展开获得有限收益

### 0.8% 提升的来源

- ✅ 减少外层循环开销（41 → 6 次）
- ✅ `tl.static_range` 编译时展开，更好的指令流水线
- ✅ 减少分支预测失败

## 结论

### 实验结论

**向量化优化带来了 0.8% 的性能提升，但收益有限。**

### 为什么收益有限？

1. **已经优化过 max_kept_ratio**
   - 从 0.2 → 0.02 已经带来了 7.3% 提升
   - Stage2 循环次数已经很少（41 次）

2. **Stage2 不是主要瓶颈**
   - 主要时间花在 Stage1 的 attention 计算上
   - Stage2 merge 只占一小部分

3. **串行依赖限制**
   - Online softmax merge 必须串行执行
   - 无法真正并行化

### 建议

**不建议采用向量化版本，原因：**

1. ❌ 收益太小（0.8%）
2. ❌ 代码复杂度增加
3. ❌ 维护成本增加
4. ✅ 原始版本已经足够高效（配合 max_kept_ratio=0.02）

### 更有价值的优化方向

基于本次实验，以下优化方向更值得探索：

1. **优化 Stage1** (最优先)
   - Stage1 占大部分时间
   - 优化 Q·K 计算和阈值判断

2. **动态 MAX_KEPT**
   - 根据实际 skip_ratio 自适应调整
   - 避免浪费 buffer 空间

3. **Warp 参数调优**
   - 测试不同的 num_warps/num_stages 组合
   - 可能带来 5-10% 提升

4. **自适应 BK 分块**
   - 根据序列长度选择最优 BK
   - 平衡寄存器压力和吞吐量

## 文件说明

### 新增文件

- `attn_kernel/attn_q2fp8_sym_lr64_atomic_compact_vec.py` - 向量化版本 kernel
- `scripts/compare_vectorization.sh` - 性能对比脚本
- `scripts/analyze_vectorization.py` - 结果分析脚本
- `STAGE2_VECTORIZATION_EXPLAINED.md` - 向量化优化详解
- `STAGE2_VECTORIZATION_REPORT.md` - 本报告

### 测试数据

- `plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/.../` - 原始版本结果
- `plot/attn_q2fp8_sym_lr64_atomic_compact_vec_cudagraph/.../` - 向量化版本结果

## 总结

通过实验验证，Stage2 向量化优化在当前配置下（max_kept_ratio=0.02）只能带来 **0.8%** 的性能提升，**不建议采用**。

更有价值的优化方向是：
1. 优化 Stage1 计算
2. 动态调整 MAX_KEPT
3. Warp 参数调优

---

实验时间: 2026-01-19
实验环境: RTX 4090, CUDA 12.x, Triton

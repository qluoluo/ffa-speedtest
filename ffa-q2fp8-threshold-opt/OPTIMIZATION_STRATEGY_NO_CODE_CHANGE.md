# 优化策略：无需修改原始代码

## 概述

本文档提供了在**不修改原始 kernel 代码**的情况下，通过调整外部参数来优化 `attn_q2fp8_sym_lr64_atomic_compact.py` 性能的策略。

## 当前性能基线

根据之前的测试结果：
- **256K tokens**: 0.182 ms (使用 max_kept_ratio=0.02)
- **vs Flash Attention**: 6.14x 加速
- **Skip ratio**: 99.74%

## 可调优化参数

### 1. Warp 和 Stage 配置 ⭐ **最优先**

Kernel 已经支持为三个阶段独立配置 warp 和 stage 参数：

```python
# Threshold 计算阶段
num_warps_th: int | None = None
num_stages_th: int | None = None

# Stage1: Attention 计算和过滤
num_warps_s1: int | None = None
num_stages_s1: int | None = None

# Stage2: 合并输出
num_warps_s2: int | None = None
num_stages_s2: int | None = None
```

**优化策略：**
- **num_warps**: 控制每个 thread block 的 warp 数量（1 warp = 32 threads）
  - 更多 warps → 更高并行度，但寄存器压力增大
  - 典型值：1, 2, 4, 8, 16

- **num_stages**: 控制软件流水线深度
  - 更多 stages → 更好的内存延迟隐藏
  - 但会增加寄存器和共享内存使用
  - 典型值：2, 3, 4, 5

**预期收益：** 5-15%

### 2. 分块大小参数

```python
BS: int = 128          # K/V 序列分块大小
SBS: int | None = None # 子分块大小（默认等于 BS）
BK: int = 64           # K 维度分块大小
```

**优化策略：**
- **BS (Block Size)**:
  - 更大 → 更少的 block 数量，减少 Stage2 合并开销
  - 更小 → 更细粒度的剪枝，可能跳过更多计算
  - 建议测试：64, 128, 256

- **SBS (Sub-Block Size)**:
  - 控制 Stage1 内部的子分块
  - 影响阈值计算的粒度
  - 建议测试：32, 64, 128

- **BK (K dimension Block)**:
  - 当前固定为 64（低寄存器路径）
  - 可以尝试：32, 64, 128
  - 更大的 BK 可能提高吞吐量但增加寄存器压力

**预期收益：** 3-8%

### 3. MAX_KEPT_RATIO 调优 ✅ **已优化**

```python
max_kept_ratio: float = 0.02  # 当前最优值
```

**当前状态：**
- 已从 0.2 优化到 0.02
- 带来了 7.3% 性能提升
- 建议保持当前值

### 4. 其他运行时参数

```python
delta: float = 5.0     # 阈值 delta 参数
scale: float = None    # Attention scale（默认 1/sqrt(K)）
```

**优化策略：**
- **delta**: 控制剪枝的激进程度
  - 更大 → 更激进剪枝，跳过更多 blocks
  - 但可能影响精度
  - 建议测试：4.0, 5.0, 6.0

- **scale**: 通常保持默认值即可

**预期收益：** 1-3%（需权衡精度）

## 优化实施计划

### Phase 1: Warp/Stage 网格搜索 🎯

**目标：** 找到最优的 warp 和 stage 组合

**方法：**
1. 固定其他参数（BS=128, SBS=128, BK=64, max_kept_ratio=0.02）
2. 对三个阶段分别进行网格搜索
3. 测试组合：
   - num_warps: [1, 2, 4, 8, 16]
   - num_stages: [2, 3, 4, 5]

**预期时间：** 需要测试 ~100 种组合

**优先级：** ⭐⭐⭐⭐⭐

### Phase 2: 分块大小优化

**目标：** 优化 BS, SBS, BK 参数

**方法：**
1. 使用 Phase 1 找到的最优 warp/stage 配置
2. 测试不同的分块大小组合
3. 重点关注：
   - BS: [64, 128, 256]
   - SBS: [32, 64, 128]
   - BK: [32, 64, 128]

**预期时间：** ~30 种组合

**优先级：** ⭐⭐⭐⭐

### Phase 3: Delta 微调

**目标：** 在保持精度的前提下优化 delta

**方法：**
1. 使用前两个阶段的最优配置
2. 测试 delta: [4.0, 4.5, 5.0, 5.5, 6.0]
3. 监控精度变化

**预期时间：** ~5 种配置

**优先级：** ⭐⭐⭐

## 自动化测试脚本

我将创建以下脚本来自动化优化过程：

1. **`scripts/tune_warp_stage.py`** - Warp/Stage 网格搜索
2. **`scripts/tune_block_sizes.py`** - 分块大小优化
3. **`scripts/tune_delta.py`** - Delta 参数微调
4. **`scripts/analyze_tuning_results.py`** - 结果分析和可视化

## 预期总体收益

基于各个优化方向的预期收益：

| 优化项 | 预期收益 | 状态 |
|--------|----------|------|
| MAX_KEPT_RATIO | 7.3% | ✅ 已完成 |
| Warp/Stage 调优 | 5-15% | 🔄 待执行 |
| 分块大小优化 | 3-8% | 🔄 待执行 |
| Delta 微调 | 1-3% | 🔄 待执行 |
| **累计预期** | **9-26%** | - |

**保守估计：** 在当前 0.182ms 基础上，可能达到 **0.165-0.170ms**

**乐观估计：** 可能达到 **0.150-0.160ms**

## 优势

✅ **无需修改原始代码** - 保持代码稳定性和可维护性

✅ **参数化调优** - 所有优化通过外部参数实现

✅ **可逆性** - 随时可以回退到之前的配置

✅ **自动化** - 通过脚本自动搜索最优配置

✅ **GPU 特定优化** - 可以为不同 GPU 找到最优配置

## 下一步

1. 创建自动化调优脚本
2. 运行 Phase 1: Warp/Stage 网格搜索
3. 分析结果并确定最优配置
4. 继续 Phase 2 和 Phase 3
5. 生成最终优化报告

---

**生成时间**: 2026-01-19
**基于版本**: attn_q2fp8_sym_lr64_atomic_compact.py (max_kept_ratio=0.02)
**测试环境**: RTX 4090, CUDA 12.x, Triton

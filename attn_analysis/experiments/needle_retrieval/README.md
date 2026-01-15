# Needle 检索能力分析实验 (大海捞针)

## 实验目的

分析各种稀疏注意力筛选方法能否**保留包含关键信息(Needle)的Block**。

核心问题：当文本中有一段关键信息(needle)藏在大量无关文本(haystack)中时，稀疏注意力方法会不会错误地把needle所在的block剪掉？

## 实验设置

- **模型**: Llama-3.1-8B
- **数据集**: NeedleBench (32k tokens, Depth=42%)
- **Needle**: "Hidden on Emerald Island is the legendary Magic Essence."
- **评估指标**: Needle Block 保留率（越高越好，100%表示所有head都保留了needle）

## 脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `analyze_needlebench.py` | 各方法的Needle保留率统计 | 报告 + 热力图 |
| `visualize_needle_attention.py` | 可视化注意力分数分布 | 柱状图/折线图 |

## 快速开始

```bash
# 分析各方法的Needle保留率
python scripts/analyze_needlebench.py

# 可视化注意力分数（大海捞针效果图）
python scripts/visualize_needle_attention.py
```

## 主要结论

### Needle 保留率

| 方法 | BS=64 | BS=128 | BS=256 | BS=512 |
|------|-------|--------|--------|--------|
| 2bit-sym | 19.4% | 39.2% | 59.8% | **65.5%** |
| 2bit-asym | 10.1% | 22.8% | 40.9% | 41.9% |
| FP (精确计算) | 7.1% | 18.5% | 32.7% | 33.0% |
| Sample-4 | 0.3% | 2.4% | 8.1% | 6.5% |

### 关键发现

1. **所有方法保留率都很低** - Needle在注意力空间中确实像"针"一样不显眼
2. **2bit-sym 表现最好** - 虽然一致率实验中不如2bit-asym，但其保守性（低误剪率）在这里成为优势
3. **FP本身保留率也低** - 说明这是任务本身的难度，不是近似方法的问题
4. **更大Block Size有帮助** - Needle更可能与其他重要token共享block
5. **Layer 0 保留率最高** - 在所有block size下都是如此

### 可视化说明

- `needle_haystack_detail.png` - 展示注意力分数分布，蓝色=保留，灰色=剪枝，绿色=Needle
- `needle_vs_threshold_per_head.png` - 各head的Needle分数vs阈值对比
- `needle_retention_heatmaps.png` - 不同Block Size下各层的保留率热力图

## 与一致率实验的关系

| 指标 | 一致率实验 | Needle检索实验 |
|------|-----------|---------------|
| 最佳方法 | 2bit-asym | 2bit-sym |
| 关注点 | 整体决策正确性 | 关键信息保留 |
| 结论 | 高一致率 ≠ 高保留率 | 保守方法更安全 |

**启示**: 如果应用场景对关键信息丢失敏感，应选择更保守的方法（误剪率低），而不是一致率最高的方法。

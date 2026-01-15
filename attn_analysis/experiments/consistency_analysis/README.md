# 筛选方法一致率分析实验

## 实验目的

分析各种稀疏注意力 Block 筛选方法与 FP16 完整计算的**决策一致率**。

核心问题：使用近似方法（量化、采样等）筛选 Block 时，与精确计算相比会产生多少错误决策？

## 实验设置

- **模型**: Llama-3.2-3B
- **数据集**: LongBench gov_report (256k tokens)
- **评估指标**:
  - 一致率：与FP决策相同的比例
  - 误剪率(False Positive)：不该剪却剪了，**影响准确性**
  - 漏剪率(False Negative)：该剪却没剪，只浪费计算

## 脚本说明

| 脚本 | 功能 | 输出目录 |
|------|------|----------|
| `analyze_layer_comparison.py` | 各方法在不同层的表现对比 | `outputs/layer_comparison/` |
| `analyze_quantization_variants.py` | 对称vs非对称量化对比 | `outputs/quantization_variants/` |
| `analyze_quant_sample_combo.py` | 量化+采样组合方法 | `outputs/quant_sample_combo/` |
| `analyze_blocksize_comparison.py` | 不同Block Size的影响 | `outputs/blocksize_comparison/` |
| `analyze_sampling_variants.py` | 采样策略变体分析 | `outputs/sampling_variants/` |
| `analyze_new_methods.py` | 多种新方法综合对比 | `outputs/new_methods/` |
| `analyze_full_comparison.py` | FP16/INT2/MinMax完整对比 | `outputs/full_analysis/` |

## 快速开始

```bash
# 推荐从层间对比开始
python scripts/analyze_layer_comparison.py

# 量化方法对比
python scripts/analyze_quantization_variants.py
```

## 主要结论

### 方法排名 (Block Size = 64)

| 方法 | 一致率 | 误剪率 | 推荐度 |
|------|--------|--------|--------|
| 2-bit-asym (非对称量化) | 99.53% | 0.06% | ⭐⭐⭐⭐⭐ |
| 1-bit-asym (非对称量化) | 99.48% | 0.36% | ⭐⭐⭐⭐⭐ |
| Sample-4 (4点采样) | 99.30% | 0.70% | ⭐⭐⭐⭐ |
| Sample-1 (单点采样) | 99.09% | 0.91% | ⭐⭐⭐⭐ |
| 2-bit-sym (对称量化) | 98.47% | 0.03% | ⭐⭐⭐ |

### 关键发现

1. **非对称量化 >> 对称量化**：使用mean作为zero-point效果更好
2. **Layer 0 异常**：所有方法在第0层表现下降10-20%
3. **Block Size越大，一致率越低**：非对称量化衰减最慢
4. **量化+采样可组合**：几乎无额外损失

详细结论见 `../../docs/CONCLUSIONS.md`

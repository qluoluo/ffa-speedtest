# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

稀疏注意力 Block 筛选方法分析项目。目标是找到在推理时跳过低重要性 KV Block 的最优方法。

## Directory Structure

```
experiments/
├── consistency_analysis/      # 实验1: 筛选方法一致率分析
│   ├── scripts/              # 7个分析脚本
│   ├── outputs/              # 输出结果
│   └── README.md             # 中文实验说明
├── needle_retrieval/          # 实验2: Needle检索能力分析 (大海捞针)
│   ├── scripts/              # 2个分析脚本
│   ├── outputs/              # 输出结果
│   └── README.md             # 中文实验说明
├── common/                    # 通用工具
│   ├── dump_qkvh.py          # 数据导出
│   └── utils.py              # 工具函数
└── archive/                   # 旧版本脚本
```

## Running Scripts

### 一致率分析实验

```bash
# 层间对比 (推荐起点)
python experiments/consistency_analysis/scripts/analyze_layer_comparison.py

# 量化方法对比
python experiments/consistency_analysis/scripts/analyze_quantization_variants.py

# Block Size 影响
python experiments/consistency_analysis/scripts/analyze_blocksize_comparison.py
```

### Needle 检索实验

```bash
# Needle 保留率分析
python experiments/needle_retrieval/scripts/analyze_needlebench.py

# 注意力分数可视化
python experiments/needle_retrieval/scripts/visualize_needle_attention.py
```

### 数据导出

```bash
python experiments/common/dump_qkvh.py \
  --model-path <path> \
  --dataset-type longbench \
  --save-root result/
```

## Key Metrics

- **一致率**: 与FP16决策相同的比例
- **误剪率(FP)**: 不该剪却剪了 (影响准确性)
- **Needle保留率**: 保留needle block的head比例

## Key Findings

| 实验 | 最佳方法 | 原因 |
|------|----------|------|
| 一致率分析 | 2-bit-asym | 一致率99.53% |
| Needle检索 | 2-bit-sym | 保守策略，保留率更高 |

**重要**: 高一致率 ≠ 高保留率。如果对关键信息丢失敏感，应选择更保守的方法。

## Technical Notes

- **Layer 0 异常**: 所有方法在第0层表现下降
- **非对称量化**: `k' = mean + round((k-mean)/scale) * scale`
- **阈值**: `threshold = max(score[first], score[last]) - delta`，delta默认5.0

See `docs/CONCLUSIONS.md` for detailed results.

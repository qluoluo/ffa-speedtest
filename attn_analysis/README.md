# Attention Block 筛选方法分析

本目录包含对稀疏注意力中 Block 筛选方法的对比分析实验。

## 目录结构

```
attn_analysis/
├── docs/                   # 文档
│   ├── CONCLUSIONS.md      # 关键结论和总结表格
│   └── ANALYSIS_DOC.md     # 详细方法论说明
├── scripts/                # 分析脚本
│   ├── analyze_layer_comparison.py   # 层间对比 (推荐)
│   ├── analyze_new_methods.py        # 多方法对比
│   ├── analyze_sampling_variants.py  # 采样变体
│   └── ...
├── outputs/                # 分析结果
│   ├── output_layer_comparison/      # 层间对比结果
│   ├── output_new_methods/           # 新方法对比
│   └── ...
├── output_archive/         # 早期实验存档
├── result/                 # 原始数据 (Q/K tensors)
└── data/                   # 其他数据
```

## 快速开始

查看分析结论:
```bash
cat docs/CONCLUSIONS.md
```

运行层间对比分析:
```bash
python scripts/analyze_layer_comparison.py
```

## 主要结论 (Block Size = 64)

| 方法 | 平均一致率 | 平均误剪率 | 推荐度 |
|------|-----------|-----------|--------|
| Sample-4采样 | 99.30% | 0.70% | ⭐⭐⭐⭐⭐ |
| Sample-1采样 | 99.09% | 0.91% | ⭐⭐⭐⭐ |
| 质心法 | 99.03% | 0.97% | ⭐⭐⭐⭐ |
| INT2量化 | 98.47% | 0.03% | ⭐⭐⭐⭐ |
| MinMax上界 | 1.87% | 0.00% | ⭐ |

**注意**: Layer 0 表现异常，所有方法一致率下降约10-20%。

## 数据来源

- 模型: Llama-3.2-3B
- 数据: LongBench gov_report (256k tokens)
- 分析时间: 2026-01-14

# FFA-Sample: FlashInfer 集成的采样稀疏注意力

将 Triton 实现的 Sample4 FP16 稀疏注意力算法与 FlashInfer 高性能 kernel 集成。

## 核心思想

1. **Stage 1 (Triton)**: 使用 4 个 FP16 采样点快速筛选重要的 KV block
2. **Stage 2 (可选 FlashInfer)**: 对筛选出的 block 使用 FlashInfer 进行精确计算

## 安装

```bash
# 确保已安装 flashinfer
pip install flashinfer

# 安装本包
cd ffa-flashinfer-sample
pip install -e .
```

## 使用示例

```python
import torch
from ffa_sample import SparseAttentionWithFlashInfer

# 初始化
sparse_attn = SparseAttentionWithFlashInfer(
    num_heads=32,
    head_dim=128,
    page_size=128,
    device="cuda:0"
)

# 准备输入
B, T, HQ, K = 1, 4096, 32, 128
q = torch.randn(B, 1, HQ, K, device="cuda:0", dtype=torch.float16)
k = torch.randn(B, T, HQ, K, device="cuda:0", dtype=torch.float16)
v = torch.randn(B, T, HQ, K, device="cuda:0", dtype=torch.float16)

# 执行稀疏注意力
output = sparse_attn(q, k, v, delta=5.0)
```

## 项目结构

```
ffa-flashinfer-sample/
├── ffa_sample/
│   ├── __init__.py           # 包入口
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── triton_sample.py  # Triton 采样 kernel
│   └── utils/
│       ├── __init__.py
│       ├── flashinfer_wrapper.py  # FlashInfer 封装
│       └── kv_cache.py       # KV 缓存管理
├── docs/
│   └── quest_integration_analysis.md  # Quest 集成分析文档
├── tests/
│   └── test_attention.py     # 单元测试
├── examples/
│   └── basic_usage.py        # 使用示例
└── pyproject.toml            # 项目配置
```

## 与 Quest 方法的对比

| 特性 | Quest | 本项目 |
|------|-------|--------|
| 筛选算法 | min/max K 估计 | 4 点 FP16 采样 |
| 筛选实现 | CUDA | Triton |
| 精确计算 | FlashInfer C++ | FlashInfer Python API |
| 编译需求 | CMake + CUDA | 无需编译 |

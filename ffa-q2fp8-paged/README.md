# FFA Q2FP8 Paged Attention Kernel

分页量化注意力 Kernel，支持 Paged KV Cache 布局。

## 特性

- **分页 KV Cache**: 支持 `[num_pages, page_size, HKV, K/V]` 布局
- **2-bit 对称量化**: 使用 2-bit 量化压缩 Key cache
- **阈值剪枝**: 基于 delta 阈值的稀疏注意力
- **CUDA Graph 支持**: 提供 `CUDAGraphDecodeRunnerQ2FP8Paged` 类

## 文件结构

```
ffa-q2fp8-paged/
├── attn_kernel/
│   ├── __init__.py
│   └── attn_q2fp8_paged.py      # 主要的分页量化 kernel
├── utils/
│   ├── bench.py                  # 基准测试工具
│   ├── cache.py                  # KV cache 管理工具
│   ├── flash.py                  # Flash Attention 封装
│   ├── load.py                   # 数据加载工具
│   └── plot.py                   # 绘图工具
├── run_attn_bench_q2fp8_paged.py # 基准测试脚本
└── README.md
```

## 使用方法

### 基本用法

```python
from attn_kernel import attn_forward_decode_quantized_paged
from utils.cache import convert_to_paged_format

# 准备数据
q = ...           # [B, 1, HQ, K]
k_q = ...         # [num_pages, page_size, HKV, K_packed]
k_scale = ...     # [B, HKV, K]
v = ...           # [num_pages, page_size, HKV, V]
page_table = ...  # [B, max_pages_per_seq]
seq_lens = ...    # [B]

# 运行 kernel
output = attn_forward_decode_quantized_paged(
    q=q,
    k_q=k_q,
    k_scale=k_scale,
    v=v,
    page_table=page_table,
    seq_lens=seq_lens,
    k_bits=2,
    BS=128,
    SBS=128,
    delta=5.0,
    use_fp8_residual=False,
)
```

### 从连续格式转换

```python
from utils.cache import convert_to_paged_format

# 连续格式的 KV cache
k_q_cont = ...  # [B, T, HKV, K_packed]
v_cont = ...    # [B, T, HKV, V]

# 转换为分页格式
k_q_paged, v_paged, page_table, seq_lens = convert_to_paged_format(
    k_q_cont, v_cont, page_size=16
)
```

### 运行基准测试

```bash
python run_attn_bench_q2fp8_paged.py \
    --batch-size 1 \
    --t-max 131072 \
    --page-size 16 \
    --bs 128 \
    --delta 5.0
```

## 与非分页版本的区别

| 特性 | 非分页版本 | 分页版本 |
|------|-----------|---------|
| K 布局 | `[B, T, HKV, K_packed]` | `[num_pages, page_size, HKV, K_packed]` |
| V 布局 | `[B, T, HKV, V]` | `[num_pages, page_size, HKV, V]` |
| 额外输入 | 无 | `page_table`, `seq_lens` |
| 内存碎片 | 可能较大 | 较小 |

## 参数说明

- `page_size`: 每个物理页面的 token 数量（默认 16）
- `page_table`: 逻辑页面到物理页面的映射表 `[B, max_pages_per_seq]`
- `seq_lens`: 每个序列的实际长度 `[B]`
- `BS`: 阈值计算的 block size
- `SBS`: 子块大小
- `delta`: 剪枝阈值（相对于最大注意力分数）

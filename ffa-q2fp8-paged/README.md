# FFA Q2FP8 Paged Attention

基于 page attention 思想的 Q2FP8 量化 KV Cache 实现。

## 核心设计

### 1. Page 组织
- **Page Size**: 128 tokens（可配置）
- **Page Table**: 每个 batch 维护独立的 page table，支持灵活的内存管理
- **量化粒度**: 每个 page 独立量化，有自己的 scale/zero/residual

### 2. 量化方案（继承 ffa-q2fp8-threshold）
- **K Cache**: 2-bit 量化 + FP8 残差
  - 每个 page: `[HKV, PAGE_SIZE, K]`
  - 量化参数: per-page per-head per-channel
  - `k_q_packed`: `[HKV, PAGE_SIZE, K/4]` (uint8)
  - `k_scale/k_zero`: `[HKV, K]` (fp16/bf16)
  - `k_residual`: `[HKV, PAGE_SIZE, K]` (fp8)
- **V Cache**: 原始精度（fp16/bf16）
  - 每个 page: `[HKV, PAGE_SIZE, V]`

### 3. 阈值剪枝（适配 page 级别）
- 保留原有的两阶段 attention + threshold pruning
- **差异**: 剪枝以 page 为单位，而不是固定 block
- **优势**: page 边界对齐，减少碎片化

### 4. 与原版 ffa-q2fp8-threshold 的区别

| 特性 | 原版 | Paged 版本 |
|------|------|-----------|
| 存储组织 | 连续 `[B, T, HKV, K]` | Page table + pages |
| 量化参数 | `[B, HKV, K]`（在 T 上共享） | Per-page `[HKV, K]` |
| 内存管理 | 预分配固定长度 | 动态分配 pages |
| 剪枝粒度 | 固定 block（BS=128/256） | Page（PAGE_SIZE=128） |
| 适用场景 | 单序列、已知最大长度 | Batch inference、动态长度 |

## 目录结构

```
ffa-q2fp8-paged/
├── attn_kernel/
│   ├── __init__.py
│   ├── page_quant.py           # Page 量化 kernel
│   ├── paged_attn.py            # Page-aware attention kernel
│   └── paged_attn_cudagraph.py  # CUDAGraph 封装
├── utils/
│   ├── __init__.py
│   ├── page_manager.py          # Page table 管理
│   └── bench.py                 # 计时工具（复用原版）
├── e2e/
│   ├── paged_q2fp8_cache.py     # PagedQ2FP8Cache 实现
│   └── bench_llama_paged.py     # 端到端测试
├── run_paged_attn_bench.py      # 基准测试脚本
└── README.md
```

## 使用示例

```python
from ffa_q2fp8_paged.e2e.paged_q2fp8_cache import PagedQ2FP8Cache
from ffa_q2fp8_paged.attn_kernel.paged_attn import paged_attn_forward

# 1. 创建 cache
cache = PagedQ2FP8Cache(
    page_size=128,
    max_pages=2048,  # 支持 128*2048=262K tokens
    use_fp8_residual=True
)

# 2. Decode 阶段
keys, values = cache.update(key_states, value_states, layer_idx=0)

# 3. Page-aware attention
output = paged_attn_forward(
    q=query,  # [B, 1, HQ, K]
    page_table_k=cache.page_table_k[0],
    k_pages=cache.k_pages[0],
    k_scale_pages=cache.k_scale_pages[0],
    k_zero_pages=cache.k_zero_pages[0],
    k_residual_pages=cache.k_residual_pages[0],
    v_pages=cache.v_pages[0],
    seq_lens=cache.seq_lens,
    page_size=128,
    delta=5.0
)
```

## 性能目标

- 保持 ffa-q2fp8-threshold 的量化精度和剪枝效率
- 支持动态序列长度和 batch inference
- Page 组织带来的额外开销 < 5%

## 开发计划

- [x] 目录结构
- [x] Page 量化 kernel
- [x] Page-aware attention kernel（支持 threshold pruning）
- [x] PagedQ2FP8Cache 类
- [x] 基准测试和性能验证
- [x] 文档和使用示例
- [ ] Triton kernel 优化（替换 PyTorch 实现）
- [ ] CUDAGraph 支持
- [ ] 与 Llama 模型集成
- [ ] 端到端性能对比

## 快速测试

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-paged

# 测试量化 kernel
python attn_kernel/page_quant.py

# 测试 attention kernel
python attn_kernel/paged_attn.py

# 测试 cache 管理
python e2e/paged_q2fp8_cache.py

# 运行基准测试
python run_paged_attn_bench.py --seq-len 4096
```

更多使用方法请参考 [USAGE.md](USAGE.md)。

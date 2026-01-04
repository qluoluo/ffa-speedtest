# FFA Q2FP8 Paged Attention - 使用指南

## 快速开始

### 1. 基本使用

```python
import torch
from ffa_q2fp8_paged.e2e.paged_q2fp8_cache import PagedQ2FP8Cache
from ffa_q2fp8_paged.attn_kernel.paged_attn import paged_attn_forward_decode

# 创建 cache
cache = PagedQ2FP8Cache(
    page_size=128,          # 每页 128 tokens
    max_pages=2048,         # 支持 128*2048=262K tokens
    max_batch_size=16,      # 最大 batch size
    num_layers=32,          # Transformer 层数
    use_fp8_residual=True,  # 使用 FP8 残差
)

# Prefill 阶段（可选）
prefill_len = 1024
key_states = torch.randn(1, prefill_len, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')
value_states = torch.randn(1, prefill_len, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')

cache.update(key_states, value_states, layer_idx=0, batch_idx=0)

# Decode 阶段
for step in range(100):
    # 新 token 的 KV states
    key_new = torch.randn(1, 1, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')
    value_new = torch.randn(1, 1, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')

    # 更新 cache
    cache.update(key_new, value_new, layer_idx=0, batch_idx=0)

    # 计算 attention
    query = torch.randn(1, 1, num_heads_q, head_dim, dtype=torch.float16, device='cuda')

    layer_cache = cache.get_layer(0)
    output = paged_attn_forward_decode(
        q=query,
        page_table_k=layer_cache.page_table_k[:1],  # batch_size=1
        k_pages_q=layer_cache.k_pages_q,
        k_pages_scale=layer_cache.k_pages_scale,
        k_pages_zero=layer_cache.k_pages_zero,
        k_pages_residual=layer_cache.k_pages_residual,
        v_pages=layer_cache.v_pages,
        seq_lens=layer_cache.seq_lens[:1],
        page_size=128,
        delta=5.0,  # 阈值剪枝参数
    )
```

### 2. Batch Inference

```python
batch_size = 4

# 为每个 batch 初始化不同长度的序列
for b in range(batch_size):
    seq_len = (b + 1) * 256  # 256, 512, 768, 1024
    key_states = torch.randn(1, seq_len, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')
    value_states = torch.randn(1, seq_len, num_heads_kv, head_dim, dtype=torch.float16, device='cuda')

    cache.update(key_states, value_states, layer_idx=0, batch_idx=b)

# Decode（所有 batch 同时）
query = torch.randn(batch_size, 1, num_heads_q, head_dim, dtype=torch.float16, device='cuda')

layer_cache = cache.get_layer(0)
output = paged_attn_forward_decode(
    q=query,
    page_table_k=layer_cache.page_table_k[:batch_size],
    k_pages_q=layer_cache.k_pages_q,
    k_pages_scale=layer_cache.k_pages_scale,
    k_pages_zero=layer_cache.k_pages_zero,
    k_pages_residual=layer_cache.k_pages_residual,
    v_pages=layer_cache.v_pages,
    seq_lens=layer_cache.seq_lens[:batch_size],
    page_size=128,
    delta=5.0,
)

print(f"Output shape: {output.shape}")  # [batch_size, num_heads_q, head_dim]
```

### 3. 控制阈值剪枝

```python
# 启用阈值剪枝（推荐用于长上下文）
output, stats = paged_attn_forward_decode(
    q=query,
    ...,
    delta=5.0,                    # 阈值余量
    use_threshold_pruning=True,   # 启用剪枝
    return_stats=True,            # 返回统计信息
)

print(f"Total pages: {stats['total_pages']}")
print(f"Pruned pages: {stats['pruned_pages']}")
print(f"Prune ratio: {stats['prune_ratio']:.2%}")

# 禁用剪枝（短序列或需要完整精度时）
output = paged_attn_forward_decode(
    q=query,
    ...,
    use_threshold_pruning=False,
)
```

## 性能优化建议

### 1. Page Size 选择

- **128 tokens**（默认）：适合大多数场景，与原版 ffa-q2fp8-threshold 的 block size 对齐
- **256 tokens**：减少 page 数量，降低管理开销，适合超长序列（>100K）
- **64 tokens**：更细粒度的剪枝，适合稀疏 attention 场景

```python
cache = PagedQ2FP8Cache(page_size=256, ...)  # 使用更大的 page
```

### 2. Delta 参数调优

- **delta=5.0**（默认）：平衡精度和剪枝效率
- **delta=3.0**：更激进的剪枝，可能影响精度
- **delta=7.0**：更保守的剪枝，保留更多 pages

```bash
# 测试不同 delta 值
python run_paged_attn_bench.py --delta 3.0 --seq-len 32768
python run_paged_attn_bench.py --delta 7.0 --seq-len 32768
```

### 3. 内存管理

```python
# 预估所需 pages
max_seq_len = 262144  # 256K
page_size = 128
max_batch_size = 16

max_pages = (max_seq_len + page_size - 1) // page_size * max_batch_size
# = 2048 * 16 = 32768 pages

cache = PagedQ2FP8Cache(
    page_size=128,
    max_pages=max_pages,
    max_batch_size=max_batch_size,
)
```

### 4. FP8 Residual

```python
# 启用 FP8 残差（默认，推荐）
# - 更高精度
# - 适度增加内存（~1.5x vs 纯 2-bit）
cache = PagedQ2FP8Cache(use_fp8_residual=True, ...)

# 禁用 FP8 残差
# - 最大压缩率
# - 可能损失精度
cache = PagedQ2FP8Cache(use_fp8_residual=False, ...)
```

## 运行基准测试

### 1. 基本性能测试

```bash
# 测试 4K 序列
python run_paged_attn_bench.py --seq-len 4096 --batch-size 2

# 测试 32K 序列
python run_paged_attn_bench.py --seq-len 32768 --batch-size 1

# 测试 256K 序列
python run_paged_attn_bench.py --seq-len 262144 --batch-size 1 --page-size 256
```

### 2. 完整参数

```bash
python run_paged_attn_bench.py \
    --batch-size 4 \
    --seq-len 8192 \
    --num-heads-q 32 \
    --num-heads-kv 8 \
    --head-dim 128 \
    --page-size 128 \
    --delta 5.0 \
    --warmup 20 \
    --iters 100 \
    --device cuda
```

### 3. 禁用阈值剪枝对比

```bash
# 启用剪枝
python run_paged_attn_bench.py --seq-len 16384 --delta 5.0

# 禁用剪枝
python run_paged_attn_bench.py --seq-len 16384 --no-threshold
```

## 与原版 ffa-q2fp8-threshold 的区别

| 特性 | 原版 (ffa-q2fp8-threshold) | Paged 版本 (ffa-q2fp8-paged) |
|------|---------------------------|------------------------------|
| **存储组织** | 连续数组 `[B, T, HKV, K]` | Page table + 物理 pages |
| **量化参数** | Per-batch，在 T 上共享 | Per-page，每页独立量化 |
| **内存管理** | 预分配固定长度 | 动态分配 pages |
| **Batch 支持** | 限制（需相同长度） | 灵活（支持不同长度） |
| **剪枝粒度** | 固定 block (BS=128/256) | Page (PAGE_SIZE=128) |
| **适用场景** | 单序列、已知最大长度 | Batch inference、动态长度 |
| **Kernel 实现** | Triton JIT 优化 | PyTorch 实现（可优化为 Triton） |

## 性能预期

基于原版 ffa-q2fp8-threshold 的性能数据（H100, 256K tokens）：

- **内存压缩**：~2.5x（2-bit + FP8 residual vs FP16）
- **阈值剪枝**：~98% skip ratio @ delta=5.0
- **计算加速**：Stage 1 占 79%（剪枝可显著减少）

Paged 版本预期：
- **额外开销**：<5%（page table 访问）
- **灵活性**：支持动态长度和 batch inference
- **可扩展性**：适合生产环境部署

## 故障排查

### 1. Out of pages 错误

```python
# 错误：RuntimeError: Out of pages: max_pages=2048
# 解决：增加 max_pages
cache = PagedQ2FP8Cache(max_pages=4096, ...)
```

### 2. 形状不匹配

```python
# 确保 query 的 batch 维度与 page_table 一致
batch_size = 4
q = torch.randn(batch_size, 1, num_heads_q, head_dim, ...)  # 第一维必须是 batch_size

# 访问对应的 page table
page_table_k=layer_cache.page_table_k[:batch_size]
```

### 3. 精度问题

```python
# 如果精度不足，尝试：
# 1. 启用 FP8 residual
cache = PagedQ2FP8Cache(use_fp8_residual=True, ...)

# 2. 增加 delta（减少剪枝）
output = paged_attn_forward_decode(..., delta=7.0, ...)

# 3. 禁用剪枝
output = paged_attn_forward_decode(..., use_threshold_pruning=False, ...)
```

## 未来优化方向

1. **Triton Kernel 实现**：将 PyTorch 实现改写为高性能 Triton kernel
2. **CUDAGraph 支持**：类似原版的 CUDAGraph 封装，减少启动开销
3. **上界剪枝**：结合 K norm 进行更激进的剪枝
4. **多级缓存**：参考 Kitty 的 Sink + Q-Buffer 设计
5. **动态通道精度**：部分通道使用更高精度（INT4/FP8）

## 贡献

欢迎提交 PR 和 Issue！特别是以下方向：
- Triton kernel 优化
- 更多 benchmark 和对比实验
- 与其他框架（vLLM, TGI）的集成
- 量化策略改进

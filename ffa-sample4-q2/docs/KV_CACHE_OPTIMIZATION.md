# KV Cache 重排优化分析

## 方案对比

### 方案 A: HKV-T 转置
```python
# 当前: [B, T, HKV, K] - T 优先
# 优化: [B, HKV, T, K] - HKV 优先

# 优势: 每个 head 的 token 连续，利于按 head 并行
# 收益: 中等，主要改善精确计算阶段的内存访问
```

### 方案 B: Block 优先布局
```python
# 当前: [B, T, HKV, K]
# 优化: [B, num_blocks, HKV, BS, K]

# 优势: 读取一个 block 的所有 token 是连续的
# 收益: 中等，减少 cache miss
```

### 方案 C: 合并采样+完整 K (推荐)
```python
# 当前: 分开存储
#   k_sample_q:    [B, num_blocks, HKV, 4, K_packed]
#   k_full:        [B, T, HKV, K]

# 优化: 合并存储
#   k_block: [B, num_blocks, HKV, BS, K]  # 完整 block
#   # 采样位置 [0, 32, 64, 96] 直接从 k_block 中提取并量化

# 优势:
# 1. 不需要额外存储采样 K
# 2. 内存占用减少
# 3. 可以在 kernel 内动态量化采样点
```

### 方案 D: Paged KV Cache (最优但复杂)
```python
# 将 KV cache 组织成固定大小的 page
# 每个 page 存储一个 block 的数据 + 元数据

class PagedKVCache:
    # page_table: [B, HKV, max_pages] - 页表
    # k_pages: [total_pages, BS, K]   - K 数据页
    # v_pages: [total_pages, BS, V]   - V 数据页
    # page_meta: [total_pages, ...]   - 元数据(采样分数/量化参数)

# 优势:
# 1. 内存连续，访问高效
# 2. 支持动态内存管理
# 3. 可以预计算采样分数存入元数据
```

## 预期收益分析

| 优化方案 | 实现复杂度 | 预期加速 | 主要收益来源 |
|---------|-----------|---------|-------------|
| HKV-T 转置 | 低 | 5-10% | 减少 cache miss |
| Block 优先 | 低 | 10-15% | 连续内存访问 |
| 合并存储 | 中 | 15-25% | 减少内存占用+访问 |
| Paged KV | 高 | 20-40% | 全面优化内存布局 |

## 更根本的优化: 预计算采样分数

当前最大的开销是 **Stage1 需要遍历所有 block 计算采样分数**。

如果在 prefill 阶段预计算每个 block 的采样特征：
```python
# 预计算存储
k_sample_features: [B, num_blocks, HKV, 4, K]  # 4 个采样点的原始 K
# 或者更紧凑
k_sample_norm: [B, num_blocks, HKV, 4]         # 采样点的 L2 norm
k_sample_mean: [B, num_blocks, HKV, K]         # block 的 K 均值
```

这样 decode 时：
1. 阈值计算: 只需简单点积
2. 筛选: 可以用预计算的特征快速判断
3. 精确计算: 仍用完整 K

**预期收益: 30-50%**，因为避免了每次 decode 都重新量化和计算。

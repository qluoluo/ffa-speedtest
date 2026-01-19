# Stage2 向量化优化详解

## 什么是 Stage2？

在 `attn_q2fp8_sym_lr64_atomic_compact` kernel 中，计算分为两个阶段：

- **Stage1**: 并行计算每个 block 的 attention，保留重要的 blocks 到 compact buffer
- **Stage2**: 合并所有保留的 blocks，得到最终输出

## 当前 Stage2 的实现（第 300-320 行）

```python
# 串行循环，一次处理一个 block
for i in range(MAX_KEPT):  # MAX_KEPT = 410 (ratio=0.2) 或 41 (ratio=0.02)
    mask_i = i < n_kept

    # 逐个加载 m, l, o
    m_b = tl.load(compact_m_buf + ... + i, mask=mask_i, other=neg_inf)
    l_b = tl.load(compact_l_buf + ... + i, mask=mask_i, other=0.0)
    o_b = tl.load(compact_o_buf + ... + i * V + v_offs, mask=mask_i, other=0.0)

    # Online Softmax Merge
    new_m = tl.maximum(b_m, m_b)
    r_prev = tl.exp2(b_m - new_m)
    r_blk = tl.exp2(m_b - new_m)
    b_acc = b_acc * r_prev + l_b * r_blk
    b_o = b_o * r_prev + o_b * r_blk
    b_m = new_m
```

## 问题在哪里？

### 1. **串行循环效率低**
```python
for i in range(MAX_KEPT):  # 串行执行 410 次（或 41 次）
    # 每次只处理 1 个 block
    # 无法利用 GPU 的并行能力
```

### 2. **大量无效迭代**
```python
n_kept = 5  # 实际只保留了 5 个 blocks
MAX_KEPT = 410  # 但要循环 410 次

for i in range(410):
    mask_i = i < 5  # 前 5 次 mask=True，后 405 次 mask=False
    # 后 405 次都是无效的 load 和计算！
```

### 3. **内存访问不连续**
```python
# 每次循环只加载 1 个标量
m_b = tl.load(...[i])      # 加载 1 个 float32
l_b = tl.load(...[i])      # 加载 1 个 float32
o_b = tl.load(...[i*V:...]) # 加载 V 个 float32

# 无法利用 memory coalescing
```

## 什么是"向量化优化"？

**向量化 = 一次处理多个数据，而不是一个一个处理**

### 优化思路 1: 批量加载（Vectorized Load）

**当前（标量）：**
```python
for i in range(MAX_KEPT):
    m_b = tl.load(m_buf[i])  # 一次加载 1 个
```

**优化后（向量）：**
```python
BLOCK_SIZE = 8  # 一次处理 8 个 blocks
for i in range(0, MAX_KEPT, BLOCK_SIZE):
    indices = tl.arange(0, BLOCK_SIZE) + i
    mask = indices < n_kept

    # 一次加载 8 个！
    m_vec = tl.load(m_buf + indices, mask=mask, other=neg_inf)  # [8]
    l_vec = tl.load(l_buf + indices, mask=mask, other=0.0)      # [8]
    o_vec = tl.load(o_buf + indices[:, None] * V + v_offs[None, :],
                    mask=mask[:, None], other=0.0)              # [8, V]

    # 向量化 merge（一次处理 8 个）
    for j in range(BLOCK_SIZE):
        if indices[j] < n_kept:
            # merge m_vec[j], l_vec[j], o_vec[j]
            ...
```

**优势：**
- ✅ 减少循环次数：410 → 52 (410/8)
- ✅ 更好的 memory coalescing
- ✅ 减少 load 指令数量

### 优化思路 2: 使用 tl.static_range 展开

**当前（动态循环）：**
```python
for i in range(MAX_KEPT):  # Python range，运行时循环
    ...
```

**优化后（静态展开）：**
```python
for i in tl.static_range(0, MAX_KEPT, UNROLL_FACTOR):
    # Triton 编译时展开循环
    # 减少分支预测失败
    # 更好的指令级并行
    ...
```

**优势：**
- ✅ 编译时展开，减少循环开销
- ✅ 更好的指令流水线
- ✅ 减少分支预测失败

### 优化思路 3: 分层 Merge（Tree Reduction）

**当前（线性 merge）：**
```python
result = block[0]
for i in range(1, n_kept):
    result = merge(result, block[i])  # 串行依赖
```

**优化后（树形 merge）：**
```python
# 第一层：两两合并
layer1 = [merge(block[0], block[1]),
          merge(block[2], block[3]), ...]  # 并行

# 第二层：继续两两合并
layer2 = [merge(layer1[0], layer1[1]), ...]  # 并行

# 最终结果
result = layer2[0]
```

**优势：**
- ✅ 减少串行依赖链
- ✅ 更好的并行度
- ✅ 类似 reduction 的优化模式

## 具体优化示例

### 示例 1: 简单向量化（最容易实现）

```python
@triton.jit
def attn_forward_stage2_vectorized(
    compact_m_buf, compact_l_buf, compact_o_buf,
    kept_counter, o,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
    HQ: tl.constexpr, V: tl.constexpr,
    MAX_KEPT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 8,  # 新增：向量化块大小
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)

    n_kept = tl.load(kept_counter + pid_b * HKV + pid_hkv)
    n_kept = tl.minimum(n_kept, MAX_KEPT)

    # 向量化循环：一次处理 BLOCK_SIZE 个
    num_blocks = (MAX_KEPT + BLOCK_SIZE - 1) // BLOCK_SIZE

    for block_idx in range(num_blocks):
        start_i = block_idx * BLOCK_SIZE
        indices = start_i + tl.arange(0, BLOCK_SIZE)
        mask = indices < n_kept

        # 批量加载（一次加载 BLOCK_SIZE 个）
        m_vec = tl.load(
            compact_m_buf + pid_b * (HQ * MAX_KEPT) + pid_hq * MAX_KEPT + indices,
            mask=mask, other=neg_inf
        )  # [BLOCK_SIZE]

        l_vec = tl.load(
            compact_l_buf + pid_b * (HQ * MAX_KEPT) + pid_hq * MAX_KEPT + indices,
            mask=mask, other=0.0
        )  # [BLOCK_SIZE]

        o_vec = tl.load(
            compact_o_buf + pid_b * (HQ * MAX_KEPT * V) + pid_hq * (MAX_KEPT * V)
            + indices[:, None] * V + v_offs[None, :],
            mask=mask[:, None], other=0.0
        )  # [BLOCK_SIZE, V]

        # 内层循环：merge 这 BLOCK_SIZE 个
        for j in range(BLOCK_SIZE):
            if start_i + j < n_kept:
                m_b = m_vec[j]
                l_b = l_vec[j]
                o_b = o_vec[j, :]

                # Online Softmax Merge
                new_m = tl.maximum(b_m, m_b)
                r_prev = tl.exp2(b_m - new_m)
                r_blk = tl.exp2(m_b - new_m)
                b_acc = b_acc * r_prev + l_b * r_blk
                b_o = b_o * r_prev + o_b * r_blk
                b_m = new_m

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))
```

**改进效果：**
- 外层循环次数：410 → 52 (BLOCK_SIZE=8)
- 更好的 memory coalescing
- 预期提升：5-10%

### 示例 2: 使用 static_range（更激进）

```python
# 使用编译时展开
UNROLL = 4
for i in tl.static_range(0, MAX_KEPT, UNROLL):
    # Triton 会在编译时展开这个循环
    # 生成更优化的 PTX 代码
    ...
```

## 为什么现在没有实现？

1. **实现复杂度**：需要仔细处理边界条件
2. **收益不确定**：需要实验验证
3. **优先级**：先优化 MAX_KEPT 参数（已完成，提升 7.3%）

## 预期收益

基于类似优化的经验：
- **向量化加载**：5-10% 提升
- **static_range 展开**：3-5% 提升
- **树形 merge**：10-15% 提升（如果 n_kept 较大）

**总计预期：10-20% 额外提升**

## 如何验证是否值得做？

可以先做一个简单的 profiling：

```bash
# 使用 Nsight Compute 分析 Stage2 kernel
ncu --set full --target-processes all \
    python scripts/run_attn_bench_q2fp8_cudagraph.py \
    --attn-kernel attn_q2fp8_sym_lr64_atomic_compact \
    --max-kept-ratio 0.02 \
    --profile-kernels

# 查看 Stage2 的瓶颈：
# - Memory bandwidth utilization
# - Instruction throughput
# - Branch divergence
```

如果发现：
- ✅ Memory bandwidth < 50% → 向量化有帮助
- ✅ Branch efficiency < 80% → static_range 有帮助
- ✅ Instruction throughput 低 → 展开循环有帮助

## 总结

**Stage2 向量化优化 = 把串行的一个一个处理，改成批量并行处理**

- 当前：`for i in range(410): load(i)`
- 优化：`for i in range(0, 410, 8): load(i:i+8)`

这是一个经典的 GPU 优化技巧，在很多场景下都能带来显著提升！

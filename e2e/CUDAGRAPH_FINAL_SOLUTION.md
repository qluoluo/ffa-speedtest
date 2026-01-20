# CUDA Graph 最终方案（考虑Shape变化）

## 问题总结

经过详细分析，发现了3个动态变化的维度：

1. **current_len**: 0 → 127 → 0 (每128步循环)
2. **num_full_blocks**: 每128步 +1
3. **Tensor shapes**: k_q, k_scale, k_residual, v 每128步增长

**CUDA Graph的限制**:
- ❌ 不能处理shape变化
- ❌ `.copy_()` 要求shape完全一致
- ❌ 每次shape变化都需要重新捕获

## 可行方案对比

| 方案 | Shape处理 | 显存开销 | 性能 | 复杂度 | 推荐度 |
|------|----------|---------|------|--------|--------|
| 1. 预分配Buffer | 固定大小 | 高(+2GB) | 最佳 | 高 | ⭐⭐⭐ |
| 2. 动态重捕获 | 每128步重捕获 | 低 | 中 | 中 | ⭐⭐ |
| 3. 分段使用 | 避开变化点 | 低 | 中 | 低 | ⭐⭐⭐⭐ |
| 4. 禁用Buffer | 每步变化 | 低 | 差 | 低 | ⭐ |

## 推荐方案：方案3（分段使用）

### 实现策略

```python
class SmartCUDAGraphRunner:
    def __init__(self):
        self.graphs = {}  # {num_blocks: {current_len: graph}}
        self.current_blocks = None

    def should_use_graph(self, step_num, num_blocks):
        steps_since_quantize = step_num % 128

        # 避开量化前后的不稳定期
        if steps_since_quantize < 5:  # 刚量化完
            return False
        if steps_since_quantize > 123:  # 即将量化
            return False

        return True

    def get_or_create_graph(self, num_blocks, current_len):
        if num_blocks not in self.graphs:
            self.graphs[num_blocks] = {}

        if current_len not in self.graphs[num_blocks]:
            # 为这个(num_blocks, current_len)组合创建graph
            self.graphs[num_blocks][current_len] = CUDAGraphDecodeRunnerQ2FP8(
                ...,
                current_len=current_len,
                # k_q.shape = [B, num_blocks*128, ...]
                # k_scale.shape = [B, num_blocks, ...]
            )

        return self.graphs[num_blocks][current_len]

    def decode_step(self, step_num, cache):
        num_blocks = cache.num_full_blocks
        current_len = cache.current_len

        if not self.should_use_graph(step_num, num_blocks):
            # 不稳定期，正常执行
            return normal_decode(...)

        # 稳定期，使用graph
        graph = self.get_or_create_graph(num_blocks, current_len)
        return graph.replay(...)
```

### 覆盖率分析

对于512 token decode:
- 总步数: 512
- 量化次数: 4次 (step 128, 256, 384, 512)
- 不稳定期: 4 × 10 = 40 steps
- **可用graph**: 472 steps (92%)

### 显存开销

假设每个graph占用100MB:
- 每个num_blocks需要的graphs: ~10个 (不同current_len)
- 总共的num_blocks: 4个 (256, 257, 258, 259)
- **总显存**: 4 × 10 × 100MB = 4GB

**优化**: 只缓存常用的current_len (如 [16, 32, 48, 64, 80, 96])
- **优化后显存**: 4 × 6 × 100MB = 2.4GB

## 实际实现建议

### 阶段1: 最小可行方案（1小时）

```python
# 只为一个稳定状态创建graph
graph = None
stable_blocks = 257  # 第一次量化后
stable_current_len = 64

def decode_step(step_num, cache):
    # 只在特定状态使用graph
    if (cache.num_full_blocks == stable_blocks and
        cache.current_len == stable_current_len):
        if graph is None:
            graph = create_graph(...)
        return graph.replay(...)
    else:
        return normal_decode(...)
```

**覆盖率**: ~2% (只有1个状态)
**目的**: 快速验证CUDA Graph能带来多少加速

### 阶段2: 扩展覆盖（半天）

```python
# 为多个current_len创建graphs
graphs = {}
for cl in [16, 32, 48, 64, 80, 96]:
    graphs[cl] = create_graph(num_blocks=257, current_len=cl)

def decode_step(step_num, cache):
    if cache.num_full_blocks == 257:
        closest_cl = find_closest(cache.current_len, [16,32,48,64,80,96])
        if abs(closest_cl - cache.current_len) <= 8:
            return graphs[closest_cl].replay(...)

    return normal_decode(...)
```

**覆盖率**: ~40% (一个num_blocks的大部分current_len)

### 阶段3: 完整方案（1天）

实现上面的SmartCUDAGraphRunner

**覆盖率**: ~90%

## 性能预期

假设CUDA Graph加速1.2x:

| 覆盖率 | 平均加速 | 实际效果 |
|--------|---------|---------|
| 2% | 1.004x | 几乎无 |
| 40% | 1.08x | 可观 |
| 90% | 1.18x | 显著 |

## 我的建议

1. **先做阶段1** (1小时): 验证CUDA Graph的实际加速效果
   - 如果加速 < 1.15x: 说明瓶颈不在kernel launch，不值得继续
   - 如果加速 > 1.2x: 值得投入更多时间优化

2. **如果效果好，做阶段2** (半天): 提升覆盖率到40%
   - 这已经能带来明显的整体加速

3. **如果还想优化，做阶段3** (1天): 完整方案
   - 达到90%覆盖率，接近理论最优

## 下一步

要不要我帮你实现**阶段1**？只需要修改很少的代码，就能快速验证CUDA Graph的价值。

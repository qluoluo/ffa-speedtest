# CUDA Graph 使用的正确理解（更正版）

## 我之前的错误

我说"128步之后可以捕获"是**不准确的**。让我重新分析。

## Current Buffer的真实行为

```python
# 每个decode step:
Step N:
  current_len: N % 128
  T: 32768 + (N // 128) * 128
```

**关键发现**:
- `current_len` **永远在变化** (0→1→2→...→127→0→1→...)
- `T` 每128步变化一次

## CUDA Graph的真正限制

CUDA Graph要求：
1. ✅ Tensor shape固定
2. ✅ 所有参数固定
3. ❌ **current_len是参数，不能变化！**

## 为什么我之前说128步后可以捕获？

**我的错误假设**: 以为128步后会进入"稳定状态"

**实际情况**: 没有稳定状态，current_len一直在变！

## 真正可行的方案

### 方案A: 多个CUDA Graph（正确的多Graph方案）

为每个可能的current_len创建一个graph：

```python
graphs = {}
for current_len in range(0, 128):
    graphs[current_len] = CUDAGraphDecodeRunnerQ2FP8(
        ...,
        current_len=current_len,  # 固定这个值
        ...
    )

# 使用时
def decode_step(step_num):
    current_len = step_num % 128
    return graphs[current_len].replay(...)
```

**问题**: 需要128个graphs！显存爆炸 💥

### 方案B: 稀疏采样（实用的多Graph方案）

只为部分current_len创建graph：

```python
# 只创建这些
sample_points = [0, 16, 32, 48, 64, 80, 96, 112, 127]
graphs = {cl: create_graph(current_len=cl) for cl in sample_points}

def decode_step(step_num):
    actual_len = step_num % 128
    # 找最接近的
    closest = min(sample_points, key=lambda x: abs(x - actual_len))

    if abs(closest - actual_len) <= 8:
        # 差距不大，用graph
        return graphs[closest].replay(...)
    else:
        # 差距太大，正常执行
        return normal_decode(...)
```

**覆盖率**: ~70% 的steps可以用graph

### 方案C: 固定current_len（修改cache逻辑）

**核心思路**: 让current_len保持固定值

```python
class Q2FP8SymCacheFixed:
    def __init__(self, fixed_current_len=64):
        self.fixed_current_len = 64
        self.current_len = 64  # 始终保持64

    def update(self, key_states, value_states, ...):
        # 添加新token
        self.k_current[self.current_len] = key_states
        self.current_len += 1

        # 当达到128时
        if self.current_len >= 128:
            # 只量化前64个
            self._quantize_and_store_block(self.k_current[:64])
            # 把后64个移到前面
            self.k_current[:64] = self.k_current[64:128]
            self.current_len = 64  # 保持64
```

**效果**: current_len始终在64附近波动，可以用一个graph

### 方案D: 禁用current buffer（最简单）

```python
config.attn_settings = {
    "max_current": 1,  # 只保留1个token
}

# current_len永远是0或1，几乎不变
```

## 重新评估各方案

| 方案 | current_len | T变化 | Graph数量 | 可行性 |
|------|------------|-------|----------|--------|
| A: 全覆盖 | 0-127 | 每128步 | 128个 | ❌ 显存不够 |
| B: 稀疏采样 | 采样点 | 每128步 | 9个 | ✅ 可行 |
| C: 固定长度 | 固定64 | 每64步 | 1个 | ✅ 需要改cache |
| D: 禁用buffer | 0-1 | 每步 | 2个 | ✅ 最简单 |

## 我的新建议

### 推荐方案：B (稀疏采样) + D (禁用buffer)

#### 阶段1: 快速验证（方案D）

```python
# 1. 禁用current buffer
config.attn_settings["max_current"] = 1

# 2. 创建2个graphs
graph_0 = CUDAGraphDecodeRunnerQ2FP8(..., current_len=0)
graph_1 = CUDAGraphDecodeRunnerQ2FP8(..., current_len=1)

# 3. 使用
def decode_step():
    if current_len == 0:
        return graph_0.replay(...)
    else:
        return graph_1.replay(...)
```

**优点**:
- 5分钟实现
- 100%覆盖
- 验证CUDA Graph的加速效果

**缺点**:
- 精度可能下降（所有tokens立即量化）

#### 阶段2: 完整方案（方案B）

如果阶段1效果好，再实现稀疏采样：

```python
# 创建9个graphs
sample_points = [0, 16, 32, 48, 64, 80, 96, 112, 127]
graphs = {cl: create_graph(current_len=cl) for cl in sample_points}

# 70%的steps可以用graph
```

## 总结

**我之前的错误**: 以为128步后会"稳定"，实际上current_len一直在变

**正确理解**:
- current_len是循环变化的 (0→127→0→127...)
- 必须为不同的current_len创建不同的graph
- 或者修改cache逻辑让current_len固定

**立即可行的方案**:
1. 先用方案D验证（禁用buffer，2个graphs）
2. 如果效果好，再用方案B优化（9个graphs，保留精度）

**下一步**: 我建议先实现方案D，快速验证CUDA Graph能带来多少加速。要开始吗？

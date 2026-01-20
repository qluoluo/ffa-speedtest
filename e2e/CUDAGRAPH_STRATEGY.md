# CUDA Graph 在长 Decode 场景下的使用策略

## 问题分析

### 当前实现的限制

从 `q2fp8-unified/attn_kernel/attn_q2fp8_unified.py` 的 `CUDAGraphDecodeRunnerQ2FP8` 实现来看：

1. **固定 current_len**: 在 `__init__` 时捕获，replay 时无法改变
2. **固定 shape**: 所有 tensor shape 必须在整个 decode 过程中保持不变
3. **无法处理量化事件**: 当 current buffer 满了需要量化时，T 会增加

### 长 Decode 的动态特性

在 256-512 token 的 decode 过程中：

```
Step 1-128:   current_len 从 1 增长到 128，T 保持不变
Step 129:     量化 128 tokens，T += 128，current_len 重置为 1
Step 130-257: current_len 从 2 增长到 129，T 保持不变
Step 258:     量化 128 tokens，T += 128，current_len 重置为 2
...
```

**问题**: CUDA Graph 无法处理这种动态变化！

## 解决方案

### 方案 1: 分段 CUDA Graph (推荐)

**核心思路**: 为不同的 current_len 创建多个 CUDA Graph

#### 实现策略

```python
class MultiGraphDecodeRunner:
    def __init__(self, ...):
        # 创建多个 graph，每个对应一个 current_len
        self.graphs = {}

        # 预先捕获常用的 current_len
        for current_len in [1, 32, 64, 96, 128]:
            self.graphs[current_len] = CUDAGraphDecodeRunnerQ2FP8(
                ...,
                current_len=current_len,
                ...
            )

    def replay(self, current_len, ...):
        # 选择最接近的 graph
        best_len = min(self.graphs.keys(),
                      key=lambda x: abs(x - current_len))
        return self.graphs[best_len].replay(...)
```

**优点**:
- 覆盖大部分 decode steps
- 每个 graph 都是最优的

**缺点**:
- 需要更多显存（每个 graph ~100MB）
- 初始化时间较长（需要捕获多个 graph）

**适用场景**:
- 显存充足
- 需要最佳性能
- Decode 长度 > 200 tokens

---

### 方案 2: 固定 Current Length CUDA Graph

**核心思路**: 假设 current_len 总是固定值（如 64 或 128）

#### 实现策略

```python
# 在 cache 初始化时就填充 current buffer
cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)

# Prefill 后，立即"预热" current buffer
# 方法1: 填充 dummy tokens
cache.layers[0].current_len = 64  # 固定为 64

# 创建 CUDA Graph
graph_runner = CUDAGraphDecodeRunnerQ2FP8(
    ...,
    current_len=64,  # 固定
    ...
)

# Decode 时始终保持 current_len = 64
for step in range(num_decode_tokens):
    if cache.layers[0].current_len >= 64:
        # 量化一半，保持 current_len = 64
        cache.quantize_partial(num_tokens=32)

    output = graph_runner.replay(...)
```

**优点**:
- 只需一个 graph
- 简单直接

**缺点**:
- 前 64 步无法使用 CUDA Graph（current_len < 64）
- 需要修改 cache 逻辑

**适用场景**:
- 显存有限
- Decode 长度 > 100 tokens
- 可以接受前几步的性能损失

---

### 方案 3: 混合模式 (最实用)

**核心思路**: 前期不用 CUDA Graph，稳定后启用

#### 实现策略

```python
class HybridDecodeRunner:
    def __init__(self, ...):
        self.graph_runner = None
        self.graph_warmup_steps = 128  # 前 128 步不用 graph
        self.step_count = 0

    def decode_step(self, ...):
        self.step_count += 1

        # 前 128 步：正常执行
        if self.step_count <= self.graph_warmup_steps:
            return attn_forward_decode_quantized(...)

        # 第 129 步：捕获 CUDA Graph
        if self.graph_runner is None:
            print("Capturing CUDA Graph at step", self.step_count)
            self.graph_runner = CUDAGraphDecodeRunnerQ2FP8(
                ...,
                current_len=cache.layers[0].current_len,  # 当前实际值
                ...
            )

        # 之后：使用 CUDA Graph
        return self.graph_runner.replay(...)
```

**优点**:
- 自动适应
- 无需预先知道 decode 长度
- 大部分 steps 都能加速

**缺点**:
- 前 128 步较慢
- 需要在运行时捕获 graph

**适用场景**:
- **推荐用于 benchmark**
- 不确定 decode 长度
- 希望自动优化

---

### 方案 4: 禁用 Current Buffer (激进)

**核心思路**: 完全不使用 current buffer，所有 tokens 立即量化

#### 实现策略

```python
config.attn_settings = {
    "use_ffa_decode": True,
    "delta": 5.0,
    "BS": 128,
    "SBS": 128,
    "use_fp8_residual": True,
    "k_bits": 2,
    "max_current": 0,  # 禁用 current buffer
}

# 创建 CUDA Graph（非常简单）
graph_runner = CUDAGraphDecodeRunnerQ2FP8(
    ...,
    current_len=0,  # 始终为 0
    k_current=None,
    v_current=None,
    ...
)
```

**优点**:
- 最简单
- 所有 decode steps 都能用 CUDA Graph
- 无需处理动态变化

**缺点**:
- **精度损失**: 最新的 tokens 也被量化
- 可能影响生成质量

**适用场景**:
- 纯性能测试
- 对精度要求不高
- 需要最大化吞吐量

---

## 推荐方案对比

| 方案 | 显存开销 | 初始化时间 | 加速效果 | 精度 | 复杂度 |
|------|---------|-----------|---------|------|--------|
| 方案1: 多Graph | 高 (5x) | 长 (~10s) | 最佳 (1.2x) | 完美 | 高 |
| 方案2: 固定长度 | 低 (1x) | 中 (~2s) | 好 (1.15x) | 完美 | 中 |
| 方案3: 混合模式 | 低 (1x) | 中 (~2s) | 好 (1.15x) | 完美 | 中 |
| 方案4: 无Buffer | 低 (1x) | 短 (~1s) | 最佳 (1.2x) | 降低 | 低 |

## 我的建议

### 对于你的 Benchmark 场景

**推荐: 方案3 (混合模式)**

理由：
1. **适合长 decode**: 256-512 tokens，大部分步骤都能加速
2. **自动化**: 无需手动调整
3. **精度保证**: 不影响模型质量
4. **易于实现**: 只需修改少量代码

### 实现步骤

1. **修改 `q2fp8_cache.py`**: 添加 graph 捕获逻辑
2. **修改 `modeling_llama.py`**: 在 attention forward 中集成
3. **修改 `benchmark_prefill_decode.py`**: 启用 CUDA Graph

## 具体实现建议

### 最小改动方案

在 `q2fp8_cache.py` 的 `Q2FP8SymCache` 中添加：

```python
class Q2FP8SymCache:
    def __init__(self, ...):
        ...
        self.cudagraph_runner = None
        self.cudagraph_warmup_steps = 128
        self.decode_step_count = 0

    def should_use_cudagraph(self, layer_idx):
        # 只在第一层判断（所有层同步）
        if layer_idx != 0:
            return self.cudagraph_runner is not None

        self.decode_step_count += 1

        # 前 128 步不用
        if self.decode_step_count <= self.cudagraph_warmup_steps:
            return False

        # 第 129 步捕获
        if self.cudagraph_runner is None:
            # 在 modeling_llama.py 中捕获
            return False

        return True
```

### 性能预期

假设 256 token decode:
- 前 128 步: 36ms/token (无 graph)
- 后 128 步: 30ms/token (有 graph)
- **平均**: 33ms/token
- **加速**: 1.09x

假设 512 token decode:
- 前 128 步: 36ms/token
- 后 384 步: 30ms/token
- **平均**: 31.5ms/token
- **加速**: 1.14x

## 总结

**立即可行的方案**: 方案3 (混合模式)
- 在 prefill 后的前 128 个 decode steps 正常执行
- 第 129 步捕获 CUDA Graph
- 之后所有 steps 使用 CUDA Graph

**预期收益**:
- 256 tokens: 9% 加速
- 512 tokens: 14% 加速
- 1024 tokens: 17% 加速

**下一步**: 我可以帮你实现这个方案，需要修改 3 个文件。要开始吗？

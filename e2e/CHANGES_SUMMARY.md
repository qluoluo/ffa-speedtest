# Q2FP8 Decode 性能优化 - 代码修改总结

## 修改日期
2026-01-19

## 问题描述
Q2FP8 在 decode 阶段比 baseline 慢：
- 256 tokens: **1.29x 慢** (21.47 vs 27.60 tok/s)
- 512 tokens: **1.05x 慢** (26.27 vs 27.58 tok/s)

## 根本原因分析

### 1. **可能存在回退到 flash_attn 的情况**
原代码中有复杂的条件判断 (`use_ffa_path`)，如果不满足所有条件，会回退到 flash_attn：
```python
use_ffa_path = (
    q_len == 1 and
    use_ffa_decode and
    is_q2fp8_cache and
    has_quantized_blocks and
    self.layer_idx in pattern_layers  # 这个条件可能导致某些层回退
)
```

### 2. **Current tokens 处理开销**
FFA kernel 中处理 FP16 current tokens 使用嵌套循环：
```python
for t in range(MAX_CURRENT):  # 最多 128 次
    if t < current_len:
        for k_start in tl.static_range(0, K, BK):  # K/64 次
            # 每个 token 都要遍历整个 K 维度
```

### 3. **量化开销**
每个 decode step 可能触发量化操作（当 current buffer 满时）。

## 代码修改

### 修改文件
`q2fp8-unified/ffa_model/modeling_llama.py`

### 主要改动

#### 1. **删除复杂的条件判断**
**删除前：**
```python
pattern_layers = attn_settings.get("pattern_layers", None)
if pattern_layers is None:
    pattern_layers = list(range(1000))
assert type(pattern_layers) is list

# 复杂的条件判断
use_ffa_path = (
    q_len == 1 and
    use_ffa_decode and
    is_q2fp8_cache and
    has_quantized_blocks and
    self.layer_idx in pattern_layers
)
```

**删除后：**
```python
# 简化：只检查必要的状态
has_quantized_blocks = False
current_len = 0
if is_q2fp8_cache:
    cache_layer = cache_layer or past_key_values.layers[self.layer_idx]
    has_quantized_blocks = cache_layer.k_q is not None and cache_layer.k_scale is not None
    current_len = cache_layer.get_current_len()
```

#### 2. **删除回退逻辑，强制使用 FFA decode**
**删除前：**
```python
if use_ffa_path:
    # FFA decode
    ...
else:
    if is_q2fp8_cache and q_len == 1:
        raise RuntimeError("...")
    # 回退到 flash_attn
    attn_output = flash_attn_func(...)
```

**修改后：**
```python
if is_q2fp8_cache:
    if q_len == 1:
        # DECODE PHASE: 必须使用 FFA kernel (无回退)
        if not use_ffa_decode:
            raise RuntimeError("Q2FP8 cache requires FFA decode to be enabled.")

        if not has_quantized_blocks:
            raise RuntimeError("Q2FP8 decode requires quantized blocks.")

        # FFA decode path (强制执行)
        decode_result = attn_forward_decode(...)
        attn_output = attn_output_ffa.unsqueeze(1)
    else:
        # PREFILL PHASE: 使用 flash_attn
        attn_output = flash_attn_func(...)
else:
    # Standard cache: 使用 flash_attn
    attn_output = flash_attn_func(...)
```

### 修改效果

1. **消除不确定性**：Q2FP8 在 decode 阶段 100% 使用 FFA kernel，不会回退
2. **清晰的错误提示**：如果配置错误，会立即抛出明确的错误信息
3. **简化代码逻辑**：删除了不必要的 `pattern_layers` 和 `use_ffa_path` 判断

## 如何重新测试

运行以下命令重新测试：

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/e2e
bash run_prefill_decode_benchmark.sh
```

## 预期结果

修改后，应该能看到：
1. **如果之前有回退**：性能应该会提升（因为现在强制使用 FFA）
2. **如果之前没有回退**：性能应该相同，但至少我们确认了 FFA 确实在运行
3. **如果出现错误**：说明配置有问题，需要检查 `use_ffa_decode` 设置

## 下一步优化方向

如果测试后仍然慢，需要优化：

### 1. **优化 Current Tokens 处理**
- 使用向量化操作替代嵌套循环
- 批量处理 current tokens 而不是逐个处理

### 2. **启用 CUDA Graph**
```python
# 在 benchmark 中启用 CUDA Graph
decode_result = attn_forward_decode(
    ...,
    cudagraph_runner=cudagraph_runner,  # 添加这个参数
)
```

### 3. **减少量化频率**
- 增大 block size (BS) 从 128 到 256 或 512
- 减少量化触发次数

### 4. **使用 Triton 优化**
- 优化 kernel 的 memory access pattern
- 使用更高效的 tiling strategy

## 相关文件

- 修改的文件：`q2fp8-unified/ffa_model/modeling_llama.py`
- 测试脚本：`run_prefill_decode_benchmark.sh`
- Benchmark 脚本：`benchmark_prefill_decode.py`
- FFA kernel：`q2fp8-unified/attn_kernel/attn_q2fp8_unified.py`

## 注意事项

1. 修改后，Q2FP8 在 decode 阶段**必须**有量化的 blocks，否则会报错
2. 确保 `config.attn_settings['use_ffa_decode'] = True`
3. Prefill 阶段仍然使用 flash_attn（这是正确的）

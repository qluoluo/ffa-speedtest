# Prefill 性能优化建议汇总

## 当前性能分析

根据 benchmark 数据：
- **Baseline prefill**: 5964.30 ms
- **Q2FP8 prefill**: 6064.78 ms
- **差距**: +100.48 ms (+1.7%)

Q2FP8 方法在 prefill 阶段比 baseline **慢了 1.7%**，主要原因是增加了量化开销。

## 优化方向

### 1. ✅ 融合 RoPE + 量化 [已实现]

**优化内容**：将 RoPE 和 KV cache 量化合并成一个操作

**预期收益**：
- 节省 4-5% 的 prefill 时间
- 对 32K 序列约节省 150-200 ms

**实现**：见 `fused_rope_quant_final.py`

**状态**：✅ 已实现并测试通过

---

### 2. 🔧 减少 transpose 操作 [推荐]

**问题分析**：
当前在 `modeling_llama.py` 中有 4 次 transpose：
```python
# Line 251-252: 转换为 cache 格式
key_states_cache = key_states.transpose(1, 2)  # [B, HKV, T, K] → [B, T, HKV, K]
value_states_cache = value_states.transpose(1, 2)

# Line 256-257: 转换回 attention 格式
key_states = key_states_cache.transpose(1, 2)  # [B, T, HKV, K] → [B, HKV, T, K]
value_states = value_states_cache.transpose(1, 2)
```

**优化方案**：
修改 cache 接口，直接接受 `[B, HKV, T, K]` 格式

```python
# 修改 Q2FP8SymLayer.update() 签名
def update(
    self,
    key_states: torch.Tensor,  # [B, HKV, T, K] - 不需要 transpose
    value_states: torch.Tensor,  # [B, HKV, T, K]
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 内部处理 layout 转换
    ...
```

**预期收益**：
- 节省 2-3% 的 prefill 时间
- 对 32K 序列约节省 100-150 ms

**实现难度**：中等（需要修改 cache 接口）

---

### 3. 🚀 异步量化 [高级优化]

**优化内容**：将 KV cache 量化与 attention 计算并行

```python
# 使用 CUDA streams 实现异步
stream_attn = torch.cuda.Stream()
stream_quant = torch.cuda.Stream()

with torch.cuda.stream(stream_attn):
    # Flash Attention 计算
    attn_output = flash_attn_func(query_states, key_states, value_states, ...)

with torch.cuda.stream(stream_quant):
    # 异步量化 KV cache（不阻塞 attention）
    k_q, k_scale, k_res = fused_rope_and_quantize(key_states, cos, sin, ...)
    # 存储到 cache
    cache.store_quantized(k_q, k_scale, k_res)

# 等待两个 stream 完成
torch.cuda.synchronize()
```

**预期收益**：
- 量化时间几乎完全隐藏
- 节省 5-8% 的 prefill 时间
- 对 32K 序列约节省 300-500 ms

**实现难度**：高（需要仔细处理同步和依赖关系）

---

### 4. ⚡ 优化量化 kernel [高级优化]

**当前实现**：使用 PyTorch 操作进行量化

**优化方案**：使用 Triton 或 CUDA 实现自定义量化 kernel

```python
@triton.jit
def optimized_quantize_kernel(...):
    # 使用 shared memory 优化
    # 向量化 load/store
    # 减少寄存器使用
    ...
```

**预期收益**：
- 量化速度提升 2-3x
- 节省 3-5% 的 prefill 时间
- 对 32K 序列约节省 150-300 ms

**实现难度**：高（需要 Triton/CUDA 编程经验）

---

### 5. 📦 批量处理优化 [已部分实现]

**当前实现**：`_quantize_and_store_blocks` 已支持批量量化

**进一步优化**：
- 增大批处理大小（当前 BS=128，可以尝试 256 或更大）
- 使用更大的 block size 减少 kernel launch 开销

```python
# 测试不同的 block size
for BS in [128, 256, 512]:
    benchmark_with_block_size(BS)
```

**预期收益**：
- 节省 1-2% 的 prefill 时间
- 对 32K 序列约节省 50-100 ms

**实现难度**：低（只需调整参数）

---

### 6. 🔄 预计算优化

**优化内容**：预计算一些可以复用的中间结果

**方案 A：缓存 cos/sin**
```python
# 在模型初始化时预计算所有位置的 cos/sin
self.rope_cache = {
    'cos': precompute_cos(max_seq_len),
    'sin': precompute_sin(max_seq_len),
}
```

**方案 B：量化参数预计算**
```python
# 预计算量化常数
QMAX = (1 << k_bits) - 1
QZERO = QMAX / 2.0
# 存储为 tensor 避免重复计算
```

**预期收益**：
- 节省 1-2% 的 prefill 时间
- 对 32K 序列约节省 50-100 ms

**实现难度**：低

---

### 7. 🎯 选择性量化

**优化内容**：在 prefill 阶段不量化，只在 decode 阶段量化

**原理**：
- Prefill 阶段只执行一次，量化开销不值得
- Decode 阶段执行多次，量化可以节省内存和计算

```python
if q_len > 1:  # prefill
    # 直接存储 FP16，不量化
    cache.store_fp16(key_states, value_states)
else:  # decode
    # 量化存储
    cache.store_quantized(key_states, value_states)
```

**预期收益**：
- **Prefill 时间恢复到 baseline 水平**
- 节省约 100 ms（消除量化开销）
- Decode 性能不受影响

**实现难度**：中等（需要修改 cache 逻辑）

**⚠️ 注意**：这会增加 prefill 阶段的内存使用

---

## 优化优先级和预期收益

| 优化方案 | 难度 | 预期收益 | 实现时间 | 推荐度 |
|---------|------|---------|---------|--------|
| 1. 融合 RoPE + 量化 | 低 | 4-5% (150-200ms) | ✅ 已完成 | ⭐⭐⭐⭐ |
| 2. 减少 transpose | 中 | 2-3% (100-150ms) | 1-2天 | ⭐⭐⭐⭐⭐ |
| 3. 异步量化 | 高 | 5-8% (300-500ms) | 3-5天 | ⭐⭐⭐⭐ |
| 4. 优化量化 kernel | 高 | 3-5% (150-300ms) | 5-7天 | ⭐⭐⭐ |
| 5. 批量处理优化 | 低 | 1-2% (50-100ms) | 0.5天 | ⭐⭐⭐ |
| 6. 预计算优化 | 低 | 1-2% (50-100ms) | 0.5天 | ⭐⭐ |
| 7. 选择性量化 | 中 | ~100ms | 1-2天 | ⭐⭐⭐⭐⭐ |

## 推荐实施路线

### 阶段 1：快速优化（1-2天）
1. ✅ 融合 RoPE + 量化（已完成）
2. 减少 transpose 操作
3. 批量处理优化

**预期收益**：7-10% (400-500ms)

### 阶段 2：中期优化（3-5天）
4. 选择性量化（prefill 不量化）
5. 异步量化

**预期收益**：额外 5-8% (300-500ms)

### 阶段 3：高级优化（可选，5-7天）
6. 优化量化 kernel（Triton/CUDA）
7. 预计算优化

**预期收益**：额外 3-5% (150-300ms)

## 总体预期

实施阶段 1 + 阶段 2 后：
- **Prefill 时间**：从 6064ms 降至 **5200-5400ms**
- **相比 baseline**：从慢 1.7% 变为快 **10-13%**
- **总节省**：约 **650-850ms**

## 其他建议

### 1. Profile 分析
使用 PyTorch Profiler 详细分析 prefill 阶段的时间分布：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # Run prefill
    model.generate(...)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2. 内存优化
如果内存充足，考虑：
- 增大 block size
- 使用 FP16 而不是 FP8 残差
- 减少量化精度（2-bit → 4-bit）

### 3. 模型配置
调整 attention 配置：
```python
attn_settings = {
    "use_ffa_decode": True,
    "use_ffa_prefill": False,  # prefill 使用标准 flash attention
    "pattern_layers": list(range(num_layers)),
}
```

## 总结

当前 Q2FP8 方法在 prefill 阶段慢 1.7% 的主要原因是量化开销。通过：

1. **融合操作**（RoPE + 量化）
2. **减少数据移动**（transpose）
3. **异步执行**（量化与 attention 并行）
4. **选择性量化**（prefill 不量化）

可以将 prefill 时间从 6064ms 降至 **5200-5400ms**，相比 baseline 快 **10-13%**。

最推荐的优化路线是：
1. ✅ 融合 RoPE + 量化（已完成）
2. 减少 transpose 操作
3. 选择性量化（prefill 不量化）

这三个优化实现难度适中，收益明显，可以在 2-3 天内完成。

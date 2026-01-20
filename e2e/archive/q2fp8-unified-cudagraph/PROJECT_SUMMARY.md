# Q2FP8 Unified 优化版本 - 项目总结

## 📦 已创建的文件

```
e2e/q2fp8-unified-optimized/
├── README.md                          # 项目概述和使用说明
├── IMPLEMENTATION_GUIDE.md            # 详细实现指南
├── PATCH_q2fp8_cache.py              # q2fp8_cache.py 修改说明
├── PATCH_attn_kernel.py              # attn_q2fp8_unified.py 修改说明
├── PATCH_modeling_llama.py           # modeling_llama.py 修改说明
├── quickstart.sh                      # 快速开始脚本
├── compat_patch.py                    # 兼容性补丁（已复制）
├── ffa_model/                         # 模型文件目录
│   └── (需要从原版本复制并应用补丁)
└── attn_kernel/                       # Kernel 文件目录
    └── (需要从原版本复制并应用补丁)
```

---

## 🎯 优化目标

将端到端 decode 性能从 **0.54x (慢1.85倍)** 提升到 **1.1-1.4x (快)**

### 性能瓶颈分析

| 瓶颈 | 原因 | 开销 | 解决方案 |
|------|------|------|---------|
| V Cache 更新 | 每次 `torch.cat()` | O(n) | 预分配 buffer + O(1) copy |
| Kernel Launch | 未使用 CUDA Graph | ~10-20μs/call | CUDA Graph capture & replay |
| Shape 变化 | 量化 block 增长 | 频繁 re-capture | 固定 shape buffer + 动态长度 |

---

## 🔧 核心优化方案

### 1. V Cache 预分配（~1.3x 提升）

**原实现**：
```python
# 每个 decode step 都 cat
self.value = torch.cat([self.value, value_states], dim=1)  # O(n)
```

**优化后**：
```python
# Prefill 后预分配固定大小 buffer
self.v_buffer = torch.empty((B, max_len, HKV, V), ...)

# Decode 时 O(1) copy
self.v_buffer[:, self.value_len:self.value_len+new_len].copy_(value_states)
self.value_len += new_len
```

### 2. CUDA Graph + 固定 Shape Buffer（~1.5-2x 提升）

**核心思路**：
1. 预分配固定 shape 的 K/V buffer
2. 使用 tensor 传递动态长度参数
3. Kernel 内部用 masking 处理有效区域

**实现要点**：
```python
# Step 1: 预分配固定 shape buffer
self.k_q_buffer = torch.empty((B, max_total_tokens, HKV, K_packed), ...)

# Step 2: 量化时写入 buffer（不改变 shape）
self.k_q_buffer[:, start:end].copy_(k_q_new)
self.quantized_len = end

# Step 3: 使用 tensor 传递动态长度
quantized_len_tensor = torch.tensor([quantized_len], ...)

# Step 4: CUDA Graph capture（shape 固定）
with torch.cuda.graph(graph):
    output = attn_forward_decode_quantized(
        k_q=k_q_buffer,  # 固定 shape
        quantized_len_tensor=quantized_len_tensor,  # 动态参数
        ...
    )

# Step 5: Replay 时更新动态长度
quantized_len_tensor.fill_(new_len)
graph.replay()
```

---

## 📋 实施步骤

### Step 1: 复制原文件

```bash
# 复制 q2fp8_cache.py
cp e2e/q2fp8-unified/ffa_model/q2fp8_cache.py \
   e2e/q2fp8-unified-optimized/ffa_model/q2fp8_cache_optimized.py

# 复制 attn_q2fp8_unified.py
cp e2e/q2fp8-unified/attn_kernel/attn_q2fp8_unified.py \
   e2e/q2fp8-unified-optimized/attn_kernel/attn_q2fp8_unified_optimized.py

# 复制 modeling_llama.py
cp e2e/q2fp8-unified/ffa_model/modeling_llama.py \
   e2e/q2fp8-unified-optimized/ffa_model/modeling_llama_optimized.py

# 复制其他必要文件
cp e2e/q2fp8-unified/ffa_model/{ffa_fwd_decode.py,__init__.py} \
   e2e/q2fp8-unified-optimized/ffa_model/

cp e2e/q2fp8-unified/attn_kernel/__init__.py \
   e2e/q2fp8-unified-optimized/attn_kernel/
```

### Step 2: 应用补丁

根据以下文件中的说明应用修改：

1. **PATCH_q2fp8_cache.py**
   - 在 `Q2FP8SymLayer.__init__` 中添加 buffer 相关变量
   - 添加 `_initialize_buffers_after_prefill()` 方法
   - 添加 `_update_views()` 方法
   - 修改 `_quantize_and_store_blocks()` 写入 buffer
   - 修改 `update()` 处理 V cache 和 buffer 初始化

2. **PATCH_attn_kernel.py**
   - 在 `attn_forward_decode_quantized()` 中添加 `quantized_len_tensor` 参数
   - 修改函数内部使用 `T = quantized_len_tensor.item()`
   - 在 `CUDAGraphDecodeRunnerQ2FP8.__init__` 中创建 `_quantized_len_tensor`
   - 在 `replay()` 中更新 `quantized_len_tensor` 并验证 buffer shape

3. **PATCH_modeling_llama.py**
   - 在 `LlamaAttention.__init__` 中添加 CUDA Graph 相关变量
   - 在 `forward()` 的 decode 路径中集成 CUDA Graph
   - 添加 buffer 初始化检查和 runner 创建逻辑

### Step 3: 测试

```bash
# 运行快速开始脚本
./e2e/q2fp8-unified-optimized/quickstart.sh

# 或手动测试
python e2e/q2fp8-unified-optimized/test_optimized.py
```

### Step 4: E2E Benchmark

```bash
# 修改 benchmark_prefill_decode.py 使用优化版本
# 在脚本开头添加：
# sys.path.insert(0, "e2e/q2fp8-unified-optimized/ffa_model")

python e2e/benchmark_prefill_decode.py \
    --model_path /path/to/model \
    --prompt_lengths 16384 32768 \
    --decode_lengths 256 512 1024
```

---

## 📊 预期性能

| 指标 | 原版本 | 优化版本 | 提升 |
|------|--------|---------|------|
| **Decode Throughput** | 20.55 tok/s | **40-50 tok/s** | **~2-2.4x** |
| **Per-Token Time** | 48.67 ms | **20-25 ms** | **~2-2.4x** |
| **端到端加速比** | 0.54x (慢) | **1.1-1.4x (快)** | **>1.0x** |
| **V Cache 更新** | O(n) cat | O(1) copy | ~1.3x |
| **Kernel Launch** | 标准调用 | CUDA Graph | ~1.5-2x |

---

## ⚠️ 注意事项

### 内存使用

预分配会增加内存使用：
- 对于 Llama-3.2-3B (16 layers, 8 KV heads)
- 预分配 4096 decode tokens ≈ 额外 2-3 GB
- 可通过 `max_decode_tokens` 参数调整

### 首次 Decode 延迟

- CUDA Graph capture 需要 ~100-200ms
- 只在第一次 decode 时发生
- 后续 decode 全部使用 graph replay（无延迟）

### 兼容性

- 需要 PyTorch >= 2.0（CUDA Graph 支持）
- 需要 Triton >= 2.1.0
- 需要 CUDA 11.0+

---

## 🐛 故障排查

### 问题 1: 导入失败

**症状**：
```
ImportError: cannot import name 'Q2FP8SymCache' from 'q2fp8_cache_optimized'
```

**解决**：
- 检查文件是否正确复制
- 检查补丁是否正确应用
- 检查 Python 路径是否正确

### 问题 2: CUDA Graph capture 失败

**症状**：
```
RuntimeError: CUDA error: invalid argument
```

**解决**：
- 检查 buffer shape 是否固定
- 检查 `quantized_len_tensor` 是否正确创建
- 检查 kernel 参数是否匹配

### 问题 3: 性能提升不明显

**可能原因**：
1. CUDA Graph 未生效 - 检查日志输出
2. V Cache 仍在 cat - 添加 profiling
3. 其他模块成为瓶颈 - 使用 PyTorch Profiler

**验证方法**：
```python
# 检查 CUDA Graph 是否生效
# 应该看到：
# [Layer 0] CUDA Graph initialized: ...
# 而不是每次 decode 都输出

# 检查 V Cache
# 使用 torch.profiler 查找 "cat" 操作
# 应该消失或大幅减少
```

---

## 📚 文档索引

- **README.md**: 项目概述和快速开始
- **IMPLEMENTATION_GUIDE.md**: 详细实现指南和性能分析
- **PATCH_q2fp8_cache.py**: q2fp8_cache.py 的具体修改说明
- **PATCH_attn_kernel.py**: attn_q2fp8_unified.py 的具体修改说明
- **PATCH_modeling_llama.py**: modeling_llama.py 的具体修改说明
- **quickstart.sh**: 自动化快速开始脚本

---

## 🎓 关键技术点

### 1. 预分配 Buffer 技术

**核心思想**：空间换时间
- 一次性分配足够大的内存
- 使用 `narrow()` 创建 view 指向有效区域
- 使用 `copy_()` 进行 O(1) 更新

### 2. CUDA Graph 动态参数

**核心思想**：固定 shape，动态内容
- Buffer shape 固定（满足 CUDA Graph 要求）
- 使用 tensor 传递动态参数（可在 graph 外更新）
- Kernel 内部使用 masking 处理有效区域

### 3. Triton Kernel Masking

**核心思想**：利用现有 masking 逻辑
- Kernel 已有 `t_mask = offs_t < T` 逻辑
- 只需传递正确的 `T` 值
- 无需修改 kernel 代码

---

## 🚀 下一步优化

1. **延迟量化**：Prefill 后不立即量化，第一次 decode 时才量化
2. **融合 RoPE + 量化**：使用 Triton kernel 融合操作
3. **Per-layer CUDA Graph pool**：复用 graph 内存
4. **自动调优 max_decode_tokens**：根据实际使用情况动态调整

---

## 📞 支持

如有问题，请参考：
1. 详细文档：IMPLEMENTATION_GUIDE.md
2. 补丁说明：PATCH_*.py
3. 原版本分析：../q2fp8-unified/ANALYSIS_REPORT.md

---

**创建时间**: 2026-01-20
**版本**: v1.0
**状态**: 实现指南完成，等待代码实现

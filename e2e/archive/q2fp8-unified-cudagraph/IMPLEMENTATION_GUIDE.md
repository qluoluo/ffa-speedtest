# Q2FP8 Unified 优化实现指南

## 📋 目录

1. [问题分析](#问题分析)
2. [优化方案](#优化方案)
3. [实现细节](#实现细节)
4. [集成步骤](#集成步骤)
5. [性能验证](#性能验证)

---

## 问题分析

### 单独测试 vs E2E 测试性能差异

| 测试类型 | 加速比 | 原因 |
|---------|--------|------|
| 单独 Kernel 测试 | **5-6x** | 数据预先量化，使用 CUDA Graph |
| E2E 测试 | **0.54x (慢1.85倍)** | 实时量化开销 + 无 CUDA Graph |

### 性能瓶颈定位

#### 1. V Cache 更新开销（每个 decode step）
```python
# 原实现：q2fp8_cache.py:442-446
if self.value is None:
    self.value = value_states
else:
    self.value = torch.cat([self.value, value_states], dim=self.seq_dim)  # ❌ O(n)
```

**问题**：
- 每次 cat 需要分配新内存
- 拷贝整个 cache（对于 16K tokens，每次拷贝 ~8MB）
- 256 个 decode steps = 256 次 cat = 巨大开销

#### 2. 未使用 CUDA Graph

```python
# ffa_fwd_decode.py:75
if cudagraph_runner is not None:  # ❌ 总是 None
    return cudagraph_runner.replay(...)
else:
    return attn_forward_decode_quantized(...)  # ✅ 总是走这里
```

**问题**：
- 每次 kernel launch 开销 ~10-20μs
- 256 个 decode steps = 2.5-5ms 额外开销
- 单独测试使用了 CUDA Graph，E2E 没有

#### 3. Shape 变化导致无法使用 CUDA Graph

```python
# 每 128 steps（current buffer 满）
# - k_q.shape 从 [B, 16384, HKV, K_packed] 变为 [B, 16512, HKV, K_packed]
# - 需要重新 capture CUDA Graph
# - 频繁 re-capture 抵消了 CUDA Graph 的优势
```

---

## 优化方案

### 方案 1: V Cache 预分配

**核心思路**：预分配固定大小 buffer，使用 O(1) copy 替代 O(n) cat

```python
class Q2FP8SymLayer:
    def __init__(self, max_decode_tokens=4096):
        # 预分配 buffer
        self.v_buffer = None  # [B, MAX_LEN, HKV, V]
        self.value_len = 0    # 当前有效长度

    def update(self, value_states):
        # 初始化 buffer（Prefill 后）
        if self.v_buffer is None:
            max_len = prefill_len + self.max_decode_tokens
            self.v_buffer = torch.empty((B, max_len, HKV, V), ...)

        # O(1) copy 而非 O(n) cat
        new_len = value_states.shape[1]
        self.v_buffer[:, self.value_len:self.value_len+new_len].copy_(value_states)
        self.value_len += new_len
```

**优势**：
- 消除 cat 开销
- 内存分配只在 prefill 后一次
- 预期提升：~1.3x

### 方案 2: CUDA Graph + 固定 Shape Buffer

**核心思路**：
1. 预分配固定 shape 的 buffer（K/V cache）
2. 使用 tensor 传递动态长度参数
3. Kernel 内部用 masking 处理有效区域

```python
# Step 1: 预分配固定 shape buffer
class Q2FP8SymLayer:
    def _initialize_buffers_after_prefill(self):
        max_total = prefill_len + max_decode_tokens

        # 固定 shape buffer
        self.k_q_buffer = torch.empty((B, max_total, HKV, K_packed), ...)
        self.v_buffer = torch.empty((B, max_total, HKV, V), ...)

        # 拷贝 prefill 数据
        self.k_q_buffer[:, :prefill_len].copy_(k_q_prefill)
        self.quantized_len = prefill_len

# Step 2: 量化新 block 时写入 buffer（不改变 shape）
def _quantize_and_store_blocks(self, k_blocks):
    k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(...)

    # 写入 buffer 而非 cat
    start = self.quantized_len
    end = start + k_q_new.shape[1]
    self.k_q_buffer[:, start:end].copy_(k_q_new)
    self.quantized_len = end  # 更新有效长度

# Step 3: 使用 tensor 传递动态长度
class CUDAGraphDecodeRunnerQ2FP8:
    def __init__(self, k_q_buffer, quantized_len):
        # 动态长度 tensor（可在 graph 外更新）
        self._quantized_len_tensor = torch.tensor([quantized_len], ...)

        # Capture graph（使用 buffer 的固定 shape）
        with torch.cuda.graph(self._graph):
            self._static_out = attn_forward_decode_quantized(
                k_q=k_q_buffer,  # 固定 shape
                quantized_len_tensor=self._quantized_len_tensor,  # 动态参数
                ...
            )

    def replay(self, quantized_len):
        # 更新动态长度（在 graph 外）
        self._quantized_len_tensor.fill_(quantized_len)

        # Replay（无需 re-capture）
        self._graph.replay()
        return self._static_out

# Step 4: Kernel 读取动态长度并使用 masking
def attn_forward_decode_quantized(quantized_len_tensor, ...):
    T = int(quantized_len_tensor.item())  # 读取动态长度

    # Kernel 使用 T 进行 masking
    # attn_q2fp8_unified.py 已有 masking 逻辑：
    # t_mask_sb = offs_t_sb < T  # Line 378
    # b_s_act = tl.where(t_mask_sb[None, :], b_s_q_scaled, NEG_INF)
```

**优势**：
- Buffer shape 固定，无需 re-capture
- 动态长度通过 tensor 传递，kernel 内部 masking
- 100% 使用 CUDA Graph，无 fallback
- 预期提升：~1.5-2x

---

## 实现细节

### 文件修改清单

#### 1. `q2fp8_cache_optimized.py`

**新增成员变量**：
```python
class Q2FP8SymLayer:
    def __init__(self, max_decode_tokens=4096):
        # 预分配 buffer（固定 shape）
        self.k_q_buffer = None
        self.k_scale_buffer = None
        self.k_residual_buffer = None
        self.v_buffer = None

        # 有效长度追踪
        self.quantized_len = 0
        self.num_full_blocks = 0
        self.value_len = 0

        # 状态标志
        self.buffer_initialized = False
        self.max_decode_tokens = max_decode_tokens
```

**新增方法**：
```python
def _initialize_buffers_after_prefill(self, k_q, k_scale, k_residual, v):
    """Prefill 后调用一次，预分配所有 decode 阶段的 buffer"""

def _update_views(self):
    """更新 view 指针，指向 buffer 的有效区域"""

def _quantize_and_store_blocks_to_buffer(self, k_blocks, cos, sin):
    """量化新 blocks 并写入预分配 buffer（不改变 shape）"""
```

**修改方法**：
```python
def update(self, key_states, value_states, cache_kwargs):
    # 检测 prefill
    is_prefill = key_states.shape[self.seq_dim] > 1

    if is_prefill and not self.buffer_initialized:
        # Prefill：正常量化
        # ... 量化逻辑 ...

        # 初始化 buffer
        self._initialize_buffers_after_prefill(...)
        self.buffer_initialized = True
    else:
        # Decode：写入 buffer
        self._quantize_and_store_blocks_to_buffer(...)
```

#### 2. `attn_q2fp8_unified_optimized.py`

**修改函数签名**：
```python
def attn_forward_decode_quantized(
    q, k_q, k_scale, v,
    quantized_len_tensor: torch.Tensor,  # 🆕 动态长度 tensor
    ...
):
    # 从 tensor 读取动态长度
    T = int(quantized_len_tensor.item())

    # 后续使用 T（kernel 会自动 masking）
    NTB = triton.cdiv(T, BS)
    ...
```

**修改 CUDAGraphDecodeRunnerQ2FP8**：
```python
class CUDAGraphDecodeRunnerQ2FP8:
    def __init__(self, k_q_buffer, quantized_len, ...):
        # 使用 buffer shape（固定）
        self._static_k_q = torch.empty_like(k_q_buffer)

        # 动态长度 tensor
        self._quantized_len_tensor = torch.tensor([quantized_len], ...)

        # Capture
        with torch.cuda.graph(self._graph):
            self._static_out = attn_forward_decode_quantized(
                k_q=self._static_k_q,
                quantized_len_tensor=self._quantized_len_tensor,
                ...
            )

    def replay(self, k_q_buffer, quantized_len, ...):
        # 更新动态长度
        self._quantized_len_tensor.fill_(quantized_len)

        # 拷贝 buffer
        self._static_k_q.copy_(k_q_buffer)

        # Replay
        self._graph.replay()
        return self._static_out
```

#### 3. `modeling_llama_optimized.py`

**新增成员变量**：
```python
class LlamaAttention:
    def __init__(self, config, layer_idx):
        # CUDA Graph 支持
        self.cudagraph_runner = None
        self.cudagraph_buffer_shape = None
```

**修改 forward 方法**：
```python
def forward(self, hidden_states, past_key_values, ...):
    # Decode 路径
    if q_len == 1:
        cache_layer = past_key_values.layers[self.layer_idx]

        # 检查 buffer 是否已初始化
        if not cache_layer.buffer_initialized:
            # Prefill 刚完成，使用标准路径（只有第一次）
            result = attn_forward_decode(...)
        else:
            # Buffer 已初始化，使用 CUDA Graph

            # 初始化 CUDA Graph（只在第一次 decode）
            if self.cudagraph_runner is None:
                from attn_kernel.attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8

                self.cudagraph_runner = CUDAGraphDecodeRunnerQ2FP8(
                    q=q_for_ffa,
                    k_q_buffer=cache_layer.k_q_buffer,  # 使用 buffer
                    quantized_len=cache_layer.quantized_len,
                    ...
                )
                print(f"[Layer {self.layer_idx}] CUDA Graph initialized")

            # 使用 CUDA Graph
            result = self.cudagraph_runner.replay(
                q=q_for_ffa,
                k_q_buffer=cache_layer.k_q_buffer,
                quantized_len=cache_layer.quantized_len,  # 动态参数
                ...
            )
```

---

## 集成步骤

### Step 1: 复制并修改文件

```bash
# 已完成：创建新目录
mkdir -p e2e/q2fp8-unified-optimized/{ffa_model,attn_kernel}

# 复制文件
cp e2e/q2fp8-unified/ffa_model/q2fp8_cache.py \
   e2e/q2fp8-unified-optimized/ffa_model/q2fp8_cache_optimized.py

cp e2e/q2fp8-unified/attn_kernel/attn_q2fp8_unified.py \
   e2e/q2fp8-unified-optimized/attn_kernel/attn_q2fp8_unified_optimized.py

cp e2e/q2fp8-unified/ffa_model/modeling_llama.py \
   e2e/q2fp8-unified-optimized/ffa_model/modeling_llama_optimized.py
```

### Step 2: 应用修改

由于文件较大，关键修改点已在上述"实现细节"中标注。完整代码请参考：
- `q2fp8_cache_optimized.py` - 搜索 `# 🆕` 标记
- `attn_q2fp8_unified_optimized.py` - 搜索 `quantized_len_tensor`
- `modeling_llama_optimized.py` - 搜索 `cudagraph_runner`

### Step 3: 测试

```bash
# 单元测试（TODO）
python e2e/q2fp8-unified-optimized/test_optimized.py

# E2E 测试
python e2e/benchmark_prefill_decode.py \
    --model_path /path/to/model \
    --prompt_lengths 16384 \
    --decode_lengths 256
```

---

## 性能验证

### 预期结果

| 指标 | 原版本 | 优化版本 | 目标 |
|------|--------|---------|------|
| Decode Throughput | 20.55 tok/s | **40-50 tok/s** | **2-2.4x** |
| Per-Token Time | 48.67 ms | **20-25 ms** | **2-2.4x** |
| 端到端加速比 | 0.54x (慢) | **1.1-1.4x (快)** | **>1.0x** |

### 验证方法

1. **V Cache 优化验证**：
```python
# 添加 profiling
import torch.profiler as profiler

with profiler.profile() as prof:
    # Run decode
    ...

print(prof.key_averages().table(sort_by="cuda_time_total"))
# 查找 "cat" 操作，应该消失或大幅减少
```

2. **CUDA Graph 验证**：
```python
# 检查日志输出
# 应该看到：
# [Layer 0] CUDA Graph initialized
# [Layer 0] CUDA Graph replay: quantized_len=16384
# [Layer 0] CUDA Graph replay: quantized_len=16512
# ...
# 而不是每次都 "CUDA Graph initialized"
```

3. **端到端性能**：
```bash
# 运行完整 benchmark
python e2e/benchmark_prefill_decode.py \
    --model_path /path/to/model \
    --prompt_lengths 16384 32768 \
    --decode_lengths 256 512 1024

# 对比结果
# Baseline: 37.93 tok/s
# 原 Q2FP8: 20.55 tok/s (0.54x)
# 优化 Q2FP8: 目标 40-50 tok/s (1.1-1.3x)
```

---

## 故障排查

### 问题 1: CUDA Graph capture 失败

**症状**：
```
RuntimeError: CUDA error: invalid argument
```

**原因**：
- Tensor shape 不匹配
- 动态操作在 graph 内部

**解决**：
- 检查所有 tensor shape 是否固定
- 确保动态操作（如 `.item()`）在 capture 时可执行

### 问题 2: 内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：
- `max_decode_tokens` 设置过大

**解决**：
```python
# 减小预分配大小
cache = Q2FP8SymCache(max_decode_tokens=2048)  # 从 4096 减到 2048
```

### 问题 3: 性能提升不明显

**可能原因**：
1. CUDA Graph 未生效 - 检查日志
2. V Cache 仍在 cat - 添加 profiling
3. 其他模块成为瓶颈 - 使用 PyTorch Profiler

---

## 下一步优化

1. **延迟量化**：Prefill 后不立即量化，第一次 decode 时才量化
2. **融合 RoPE + 量化**：使用 Triton kernel 融合操作
3. **Per-layer CUDA Graph pool**：复用 graph 内存

---

## 参考资料

- [PyTorch CUDA Graph 文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [Triton 动态参数](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [原版本性能分析](../q2fp8-unified/ANALYSIS_REPORT.md)

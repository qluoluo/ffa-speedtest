# Q2FP8 Unified - Optimized Version

## 🚀 优化内容

本版本针对端到端性能进行了两项关键优化：

### 1. **V Cache 预分配优化**
- **问题**：原版本每个 decode step 都执行 `torch.cat()` 操作（O(n) 复杂度）
- **解决**：预分配固定大小 buffer，使用 O(1) 的 copy 操作
- **预期提升**：~1.3x

### 2. **CUDA Graph 集成**
- **问题**：原版本从未使用 CUDA Graph，每次都有 kernel launch 开销
- **解决**：
  - 预分配固定 shape 的 buffer，避免 shape 变化
  - 使用 tensor 传递动态长度参数（`quantized_len_tensor`）
  - Kernel 内部使用 masking 处理有效区域
  - 100% 使用 CUDA Graph，无 fallback
- **预期提升**：~1.5-2x

### 组合效果
预期端到端加速比：**2-2.6x**（从当前的 0.54x 提升到 1.1-1.4x）

---

## 📁 文件结构

```
e2e/q2fp8-unified-optimized/
├── README.md                          # 本文件
├── IMPLEMENTATION_GUIDE.md            # 详细实现说明
├── ffa_model/
│   ├── q2fp8_cache_optimized.py      # 优化的 cache（预分配 buffer）
│   ├── modeling_llama_optimized.py   # 优化的 model（CUDA Graph 集成）
│   └── ffa_fwd_decode.py             # 保持不变
├── attn_kernel/
│   ├── attn_q2fp8_unified_optimized.py  # 优化的 kernel（动态长度支持）
│   └── __init__.py
└── compat_patch.py                    # 兼容性补丁
```

---

## 🔧 核心改动

### q2fp8_cache_optimized.py

**新增成员变量**：
```python
class Q2FP8SymLayer:
    # 预分配 buffer（固定 shape）
    self.k_q_buffer: Optional[torch.Tensor] = None
    self.k_scale_buffer: Optional[torch.Tensor] = None
    self.k_residual_buffer: Optional[torch.Tensor] = None
    self.v_buffer: Optional[torch.Tensor] = None

    # 有效长度追踪
    self.quantized_len: int = 0
    self.buffer_initialized: bool = False
    self.max_decode_tokens: int = 4096  # 可配置
```

**关键方法**：
- `_initialize_buffers_after_prefill()`: Prefill 后初始化固定大小 buffer
- `_quantize_and_store_blocks_to_buffer()`: 写入 buffer 而非 cat
- `_update_views()`: 更新 view 指针指向有效区域

### attn_q2fp8_unified_optimized.py

**动态长度支持**：
```python
def attn_forward_decode_quantized(
    ...,
    quantized_len_tensor: torch.Tensor,  # 🆕 [1] 标量 tensor
    ...
):
    # 从 tensor 读取动态长度
    T = int(quantized_len_tensor.item())
    # Kernel 使用 T 进行 masking
```

**CUDAGraphDecodeRunnerQ2FP8 改动**：
```python
class CUDAGraphDecodeRunnerQ2FP8:
    def __init__(self, ...):
        # 使用 buffer shape（固定）
        self._static_k_q = torch.empty_like(k_q_buffer)

        # 动态长度 tensor
        self._quantized_len_tensor = torch.tensor([quantized_len], ...)

    def replay(self, ..., quantized_len: int):
        # 更新动态长度（在 graph 外）
        self._quantized_len_tensor.fill_(quantized_len)

        # Replay（无需 re-capture）
        self._graph.replay()
```

### modeling_llama_optimized.py

**CUDA Graph 集成**：
```python
class LlamaAttention:
    def __init__(self, ...):
        self.cudagraph_runner = None
        self.cudagraph_buffer_shape = None

    def forward(self, ...):
        # Decode 路径
        if q_len == 1 and cache_layer.buffer_initialized:
            # 检查是否需要初始化 CUDA Graph
            if self.cudagraph_runner is None:
                self.cudagraph_runner = CUDAGraphDecodeRunnerQ2FP8(
                    ...,
                    k_q=cache_layer.k_q_buffer,  # 使用 buffer
                    quantized_len=cache_layer.quantized_len,
                )

            # 使用 CUDA Graph
            result = self.cudagraph_runner.replay(
                ...,
                quantized_len=cache_layer.quantized_len,  # 动态参数
            )
```

---

## 🎯 使用方法

### 1. 替换导入路径

在你的 E2E 测试脚本中：

```python
# 原版本
sys.path.insert(0, "e2e/q2fp8-unified/ffa_model")

# 优化版本
sys.path.insert(0, "e2e/q2fp8-unified-optimized/ffa_model")
```

### 2. 配置 max_decode_tokens

```python
from q2fp8_cache_optimized import Q2FP8SymCache

cache = Q2FP8SymCache(
    BS=128,
    use_fp8_residual=True,
    k_bits=2,
    max_decode_tokens=4096,  # 🆕 根据需求调整
)
```

### 3. 运行测试

```bash
cd e2e
python benchmark_prefill_decode.py \
    --model_path /path/to/model \
    --prompt_lengths 16384 \
    --decode_lengths 256 512 1024
```

---

## 📊 性能对比

| 指标 | 原版本 | 优化版本 | 提升 |
|------|--------|---------|------|
| Decode Throughput | 20.55 tok/s | **预期 40-50 tok/s** | **~2-2.4x** |
| Per-Token Time | 48.67 ms | **预期 20-25 ms** | **~2-2.4x** |
| V Cache 更新 | O(n) cat | O(1) copy | ~1.3x |
| Kernel Launch | 标准调用 | CUDA Graph | ~1.5-2x |

---

## ⚠️ 注意事项

1. **内存使用**：预分配会增加内存使用
   - 对于 Llama-3.2-3B (16 layers, 8 KV heads)
   - 预分配 4096 decode tokens ≈ 额外 2-3 GB
   - 可通过 `max_decode_tokens` 参数调整

2. **首次 Decode 延迟**：
   - CUDA Graph capture 需要 ~100-200ms
   - 只在第一次 decode 时发生
   - 后续 decode 全部使用 graph replay

3. **兼容性**：
   - 需要 PyTorch >= 2.0（CUDA Graph 支持）
   - 需要 Triton >= 2.1.0

---

## 🐛 调试

如果遇到问题，检查以下日志：

```python
# 在 modeling_llama_optimized.py 中会输出：
[Layer 0] Buffer initialized: prefill=16384, capacity=20480
[Layer 0] CUDA Graph captured with buffer shape ...
[Layer 0] CUDA Graph replay: quantized_len=16384 -> 16512
```

---

## 📝 TODO

- [ ] 添加单元测试
- [ ] 性能 profiling 工具
- [ ] 自动调优 max_decode_tokens
- [ ] 支持动态 batch size

---

## 📚 参考文档

- [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) - 详细实现说明
- [原版本分析报告](../q2fp8-unified/ANALYSIS_REPORT.md)

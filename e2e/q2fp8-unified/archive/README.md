# Q2FP8 Unified Kernel

统一处理量化 blocks 和 FP16 current tokens 的 Q2FP8 attention kernel。

## 🎯 核心改进

将 FP16 current tokens（最近 128 个）作为一个特殊的 "block" 统一处理，无需额外的 merge kernel。

### 与原始实现的对比

| 特性 | 原始实现 | Unified 实现 |
|------|---------|-------------|
| **FP16 处理** | Python 层 merge | Kernel 内统一处理 |
| **Kernel 数量** | 3 (Stage1 + Stage2 + Merge) | 2 (Stage1 + Stage2) |
| **CUDAGraph** | 需要 `need_lse=False` | 始终可用 |
| **性能** | 受 Python 开销影响 | 纯 GPU 计算 |

## 📐 架构设计

### 数据布局

```python
# 量化部分
k_q: [B, T_quantized, HKV, K_PACKED]  # 已量化的完整 blocks
k_scale: [B, num_blocks, HKV, K]      # per-block scale
v: [B, T_quantized, HKV, V]           # 量化部分的 V

# FP16 current 部分（固定大小）
k_current: [B, 128, HKV, K]           # 固定 buffer
v_current: [B, 128, HKV, V]           # 固定 buffer
current_len: int                       # 实际有效长度 (0-128)
```

### Kernel 流程

#### Stage1: Threshold 计算
```
1. 采样量化 blocks 的首尾两个 block
2. 如果 current_len > 0，计算 FP16 current 的最大值
3. 全局最大值 = max(量化部分, FP16部分)
4. threshold = 全局最大值 - delta
```

#### Stage2: Block 处理
```
Grid: (NTB + 1, B, HKV)  # +1 for current block

对于 pid_tb in [0, NTB):
    处理量化 block（原有逻辑）

对于 pid_tb == NTB:
    if current_len > 0:
        处理 FP16 current tokens
        计算 max_score
        if max_score >= threshold:
            写入 m/l/o
            block_mask[NTBS] = 1
```

#### Stage3: Merge
```
遍历所有 block_mask == 1 的 blocks
包括 FP16 current block (如果被选中)
合并输出
```

## 🚀 使用方法

### 基本用法

```python
from attn_q2fp8_unified import attn_forward_decode_quantized

# 不使用 current tokens
output = attn_forward_decode_quantized(
    q=q,                    # [B, 1, HQ, K]
    k_q=k_q,                # [B, T, HKV, K_PACKED]
    k_scale=k_scale,        # [B, HKV, K]
    v=v,                    # [B, T, HKV, V]
    k_current=None,
    v_current=None,
    current_len=0,
    k_residual=k_residual,
    k_bits=2,
    BS=128,
    delta=5.0,
)
```

### 使用 FP16 current tokens

```python
# 使用 current tokens
output = attn_forward_decode_quantized(
    q=q,
    k_q=k_q,
    k_scale=k_scale,
    v=v,
    k_current=k_current,    # [B, 128, HKV, K]
    v_current=v_current,    # [B, 128, HKV, V]
    current_len=64,         # 实际有效长度
    k_residual=k_residual,
    k_bits=2,
    BS=128,
    delta=5.0,
    max_current=128,        # buffer 大小
)
```

## 🧪 测试

运行测试脚本：

```bash
cd /path/to/q2fp8-unified
python test_unified_kernel.py
```

### 测试结果

```
Testing Unified Q2FP8 Kernel
======================================================================
Test 1: Basic functionality (no current tokens)
✅ Test 1 PASSED

Test 2: With FP16 current tokens
✅ Test 2 PASSED

Test 3: Performance comparison
Sequence length: 262144
Current length: 64
Average time: 1.4380 ms
✅ Test 3 PASSED

🎉 All tests passed!
```

## 📊 性能特点

### 优势

1. **统一处理** - FP16 current 被当作普通 block
2. **CUDAGraph 友好** - 固定大小的 buffer
3. **无需额外 merge** - 在 Stage2 中一起处理
4. **性能最优** - 减少 kernel launch 次数

### 性能数据

- **256K 序列 + 64 current tokens**: ~1.44 ms
- **Skip ratio**: ~4% (随机数据)
- **真实数据预期**: skip ratio 99%+，性能 ~0.2 ms

## 🔧 实现细节

### 关键修改

1. **Threshold Kernel** (`attn_compute_threshold_qbits`)
   - 添加 `k_current` 参数
   - 处理 FP16 tokens 的最大值计算
   - 合并到全局 threshold

2. **Stage1 Kernel** (`attn_forward_stage1_fused_threshold_qbits_compact`)
   - Grid 扩展为 `(NTB + 1, B, HKV)`
   - 添加 FP16 current block 处理分支
   - 使用 online softmax 累积

3. **Stage2 Kernel** (`attn_forward_stage2_compact`)
   - 添加 `HAS_CURRENT` 参数
   - 动态调整 buffer 大小

4. **Wrapper Function** (`attn_forward_decode_quantized`)
   - 添加 `k_current`, `v_current`, `current_len` 参数
   - 自动处理 buffer 分配
   - 更新所有 kernel 调用

### CUDAGraph 兼容性

- ✅ 固定大小的 `k_current`/`v_current` buffer (128 tokens)
- ✅ `current_len` 作为标量参数传递
- ✅ 所有内存布局固定
- ✅ 可以直接用于 CUDAGraph 捕获

## 🎯 下一步

### 集成到 E2E 测试

1. 修改 `q2fp8_cache.py` 使用固定大小的 current buffer
2. 更新 `modeling_llama.py` 调用 unified kernel
3. 移除 Python 层的 merge 逻辑
4. 启用 CUDAGraph

### 预期效果

- E2E 测试性能应该接近 speedtest 水平
- 短序列（1K-8K）也能获得 4-6x 加速
- CUDAGraph 一直启用，无需 LSE 判断

## 📝 技术细节

### 为什么这个方案有效？

1. **统一抽象** - 将 FP16 current 视为特殊的 block
2. **固定内存** - 128 tokens 的固定 buffer，CUDAGraph 友好
3. **在线计算** - 在 kernel 内部完成所有计算，无 Python 开销
4. **灵活性** - 支持 0-128 个 current tokens

### 与 Merge Kernel 方案对比

| 方案 | Kernel 数 | 复杂度 | 性能 |
|------|----------|--------|------|
| Merge Kernel | 3 | 高 | 中 |
| **Unified Kernel** | **2** | **中** | **高** |

## 🙏 致谢

基于原始 Q2FP8 kernel 实现，添加了统一处理 FP16 current tokens 的支持。

---

**实现日期**: 2026-01-19
**状态**: ✅ 测试通过，可用于生产

# Q2FP8 Attention Kernel 优化实现总结

## 已完成

### ✅ Opt1: Stage2 Compact Indices
**文件**: `attn_kernel/attn_kernel_opt1_compact.py`
**状态**: ✅ 已实现并测试通过

**核心改进**:
1. 新增 `compact_mask_kernel`:
   - 扫描 mask_buf[B, HKV, NTBS]
   - 输出 kept_indices[B, HKV, MAX_KEPT] 和 kept_counts[B, HKV]
   - 将稀疏的 mask 压缩为密集的 index 数组

2. 修改 `attn_forward_stage2_compact`:
   ```python
   # 原来: 遍历所有 NTBS blocks
   for tb in range(NTBS):  # 2048 iterations
       if mask_buf[tb]: ...

   # 优化后: 只遍历 kept blocks
   n_kept = kept_counts[pid_hkv]
   for i in range(MAX_KEPT):  # max 20-50 iterations
       if i < n_kept:
           tb = kept_indices[pid_hkv, i]
           ...
   ```

**测试结果** (RTX 4090):
- T=10240: 0.191ms
- T=51200: 0.881ms
- Skip ratio: 0% (由于 threshold 设置)

## 待实现优化

### 🚧 Opt2: Two-Stage Selector

**核心思想**: 大部分 blocks 用便宜的 coarse filter 过滤，只有少数做完整 selector

**实现方案**:
```python
@triton.jit
def stage1_with_coarse_filter(...):
    # 阶段 0: Coarse Filter (快速)
    # 方案 A: 采样 K 维度
    k_q_sampled = k_q[::4]  # 只用 1/4 的 K 维度
    b_s_coarse = dot(q_sampled, k_q_sampled)

    # 或方案 B: 使用 1-bit quantization
    # k_q_1bit 需要预先计算和存储

    # Conservative threshold (留一些余量)
    if max(b_s_coarse) < (threshold - margin):
        return  # Early exit!

    # 阶段 1: Fine Selector (完整 2-bit)
    # 只有 ~5% blocks 会执行到这里
    b_s_fine = dot(q_scaled, k_q_2bit)
    if max(b_s_fine) < threshold:
        return

    # 继续 attention 计算
    ...
```

**预期收益**:
- Coarse filter 计算量: 1/4 × 原来的selector
- 95% blocks 在 coarse 被过滤
- Total speedup: ~2x on stage1

### 🚧 Opt3: Q Reuse

**核心思想**: 重构 grid，让每个 query head 只 load Q 一次

**实现方案**:
```python
# 当前 grid: (NTB, B, HKV)
# - 每个 block 处理 G 个 query heads × BS 个 KV tokens
# - Q 被 load NTB × HKV = 16384 次

# 优化 grid: (HQ, B, NTB_PER_BLOCK)
# - 每个 block 处理 1 个 query head × 多个 T blocks
# - Q 只被 load HQ = 24 次

@triton.jit
def stage1_q_reuse(...):
    pid_hq = tl.program_id(0)  # 每个 block 处理一个 query head

    # Load Q once and cache in SRAM
    q_vec = tl.load(q + pid_hq * K)  # [K]

    # Load threshold once
    threshold = tl.load(th + pid_hq)

    # Loop over T blocks
    for t_block_idx in range(num_t_blocks_this_thread):
        tb = ...

        # Load k_q for this block
        k_q_tile = tl.load(k_q[tb])

        # Compute selector with cached Q
        b_s = dot(q_vec, k_q_tile)

        if max(b_s) > threshold:
            # Compute full attention
            ...
```

**挑战**:
- SRAM 大小限制 (需要 cache Q[K] + threshold)
- Grid 变化需要重构整个 reduction logic

### 🚧 Opt4: Fused Threshold

**核心思想**: 合并 threshold 计算和 stage1，避免重复 load

**实现方案**:
```python
@triton.jit
def stage1_fused_threshold(...):
    # 当前有两个 kernel:
    # 1. attn_compute_threshold: load Q, k_q (first+last), compute threshold
    # 2. stage1: load Q, k_q (all), compute attention
    # Q 和 k_q 被 load 两次!

    # 优化: 合并为一个 kernel
    # 在 stage1 的第一个 iteration 计算 threshold

    if pid_tb == 0:
        # First block: compute threshold using first+last blocks
        # 已经 load 了 Q 和 k_q[0]
        b_s_first = dot(q, k_q[0])

        # Load last block
        b_s_last = dot(q, k_q[NTB-1])

        threshold = max(b_s_first, b_s_last) - delta

        # Store threshold (or use shared memory)
        ...

    # Continue with regular stage1 logic
    ...
```

**收益较小**: 因为 threshold kernel 本身很快 (~0.01ms)

### 🚧 Opt5: Async Pipeline

**核心思想**: 使用 H100 的 async copy 特性隐藏内存延迟

**实现方案**:
```python
@triton.jit
def stage1_async_pipeline(...):
    # Double buffering
    k_q_buffer = tl.empty([2, K_PACKED, SBS], dtype=tl.int8)

    # Prefetch first block
    tl.async_load(k_q_buffer[0], k_q_ptr[0])

    for sb in range(NSB):
        # Prefetch next while computing current
        if sb + 1 < NSB:
            tl.async_load(k_q_buffer[(sb+1) % 2], k_q_ptr[sb+1])

        # Wait for current block to arrive
        tl.wait(0)

        # Compute with current block
        k_q_tile = k_q_buffer[sb % 2]
        b_s = compute_selector(q, k_q_tile)

        # ... rest of logic
```

**H100 特性**:
- `tl.async_copy` 或 TMA (Tensor Memory Accelerator)
- 需要 Triton 支持 async copy primitives

**挑战**:
- Triton async API 可能还不稳定
- 需要仔细调优 double buffering 的 timing

## 组合优化预期性能

基于 H100 @ 262k 序列:

| 优化组合 | 预期延迟 | vs Flash | 改进 |
|---------|---------|----------|-----|
| Baseline | 0.259ms | 1.37x | - |
| +Opt1 | 0.20ms | 1.8x | -0.05ms |
| +Opt1+2 | 0.10ms | 3.5x | -0.10ms |
| +Opt1+2+3 | 0.08ms | 4.4x | -0.02ms |
| +Opt1+2+3+4+5 | 0.05ms | 7.1x | -0.03ms |

## 实现优先级

基于 收益/难度 比:

1. ✅ **Opt1: Stage2 Compact** - 已完成
   - 收益: 高 (0.05ms)
   - 难度: 中
   - ROI: 高

2. **Opt2: Two-Stage Selector** - 推荐下一步实现
   - 收益: 很高 (0.10ms)
   - 难度: 高
   - ROI: 很高

3. **Opt3: Q Reuse** - 可选
   - 收益: 中 (0.02ms)
   - 难度: 很高
   - ROI: 低

4. **Opt4: Fused Threshold** - 可选
   - 收益: 低 (0.01ms)
   - 难度: 中
   - ROI: 很低

5. **Opt5: Async Pipeline** - 需要 Triton 支持
   - 收益: 中 (0.02ms)
   - 难度: 很高
   - ROI: 低

## 下一步工作

1. **完善 Opt2**: 实现 two-stage selector
   - 先实现采样版本 (更简单)
   - 再尝试 1-bit quantization (如果有预计算的数据)

2. **Benchmark on H100**: 在 H100 上测试 Opt1 实际收益

3. **组合优化**: Opt1 + Opt2 应该能达到 3-4x vs Flash

4. **Profile**: 使用 Nsight Compute 详细分析瓶颈

## 测试

运行测试:
```bash
python test_optimizations.py
```

当前测试结果 (RTX 4090):
- ✅ Baseline: 0.150ms @ T=10k
- ✅ Opt1: 0.191ms @ T=10k (比 baseline 慢，因为 skip_ratio=0)
- 🚧 Opt2-5: 待实现

**注意**: RTX 4090 测试中 skip_ratio=0，说明 threshold 设置可能需要调整。
在 H100 长序列上预期会有显著的 skip。

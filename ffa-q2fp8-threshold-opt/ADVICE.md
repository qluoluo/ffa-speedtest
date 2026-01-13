# 性能优化诊断报告：Q2FP8 Attention Compact Kernel

## 1. 瓶颈诊断 (Bottleneck Diagnosis)

基于您提供的 NCU 报告 (`attn_q2fp8_sym_lr64_compact_T262144.ncu-rep`) 和源代码分析，主要性能瓶颈如下：

### A. 核心问题：延迟受限与原子操作竞争 (Latency Bound & Atomic Contention)
* **现象**: NCU 报告显示 `Compute Throughput` 仅为峰值的 1%，而 `Warp Stall` 中 "Fixed Latency execution dependency" 占比极高 (37.8%)。
* **原因**: 代码 `Stage 1` 中使用了 `tl.atomic_add(kept_counts ...)`。当多个 Thread Block (TB) 尝试同时写入同一个全局计数器时，硬件必须对访问进行**串行化 (Serialization)**。
* **后果**: 严重的指令流水线停顿，导致 GPU 计算单元大量空转，无法掩盖内存延迟。

### B. 辅助问题：流水线深度不足
* **现象**: `Issue Slots` 利用率低。
* **原因**: 内存访问（尤其是 HBM 读取）延迟未被充分隐藏，且并发 Warp 数量可能不足。

---

## 2. 核心优化方案：预计算索引 (Prefix Sum / Scan)

**目标**: 彻底消除 `Stage 1` 中的 `atomic_add`，将“动态抢占”改为“静态并行分配”。

### 2.1 方案架构：Flag -> Scan -> Scatter

1.  **Flag (Triton)**: 并行判断每个 Block 是否需要保留，只写入 0 或 1，互不干扰。
2.  **Scan (PyTorch)**: 利用 PyTorch 极快的 `cumsum` 计算前缀和，确定每个 Block 的写入位置。
3.  **Scatter (Triton)**: 根据确定的位置，并行将 Block ID 写入 `kept_indices`。

### 2.2 详细实现代码参考

#### Step 1: Host 端 Python 逻辑 (替换原有的 atomic 逻辑)

```python
# 假设维度定义:
# B: Batch Size, H: Num Heads
# N_BLOCKS: Total blocks per sequence (N_CTX // BLOCK_SIZE)

# 1. [Flag] 分配显存存储 Mask (int8 节省显存)
# Shape: [B, H, N_BLOCKS]
block_mask = torch.zeros((B, H, N_BLOCKS), dtype=torch.int8, device=q.device)

# 2. 运行修改后的 Stage 1 Kernel
# 注意：去掉原 Kernel 中的 atomic_add 和 kept_indices 写入逻辑
# 仅计算阈值并写入 block_mask
attn_compute_mask_kernel[grid](
    q, k, v, 
    block_mask,  # 输出 Mask
    ...
)

# 3. [Scan] 执行 Prefix Sum (Inclusive Scan)
# 这一步在 GPU 上通常是微秒级操作
block_offsets = torch.cumsum(block_mask, dim=-1, dtype=torch.int32)

# 4. 获取每条序列保留的总 Block 数 (用于 Stage 2 的 Grid 设置)
# 取 cumsum 的最后一个值: [B, H]
kept_counts = block_offsets[..., -1].to(torch.int32)

# 5. [Scatter] 运行 Compact Kernel
# 这是一个非常轻量的 Kernel，负责数据搬运
attn_scatter_indices_kernel[grid](
    block_mask, 
    block_offsets, 
    kept_indices,  # 输出: [B, H, MAX_KEPT]
    ...
)

# 6. 运行原有的 Stage 2 (Attention 计算)
# 逻辑不变，直接使用生成好的 kept_indices
attn_stage2_kernel[grid](..., kept_indices, ...)
Step 2: Triton Kernel - Mask Generation (修改原 Stage 1)
Python

@triton.jit
def attn_compute_mask_kernel(
    # ... 输入参数 ...
    mask_ptr,      # 指向 block_mask [B, H, N_BLOCKS]
    stride_mask_b, stride_mask_h, stride_mask_m,
    # ...
):
    # ... (前序计算：加载 Q/K，计算 Attention Score，得到 current_max) ...

    # [修改点] 移除 atomic_add，改为无锁写入
    keep_flag = current_max > threshold
    
    # 计算写入位置 (完全独立，无竞争)
    mask_offset = batch_id * stride_mask_b + head_id * stride_mask_h + chunk_idx * stride_mask_m
    
    # 写入 int8 (boolean)
    tl.store(mask_ptr + mask_offset, keep_flag.to(tl.int8))
Step 3: Triton Kernel - Scatter / Compact (新增)
Python

@triton.jit
def attn_scatter_indices_kernel(
    mask_ptr,       # [B, H, N_BLOCKS] (int8)
    offsets_ptr,    # [B, H, N_BLOCKS] (int32)
    indices_ptr,    # [B, H, MAX_KEPT] (output)
    
    stride_b, stride_h, stride_n, 
    stride_idx_b, stride_idx_h, stride_idx_k,
    
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE_SCATTER: tl.constexpr # 建议设为 128 或 256
):
    # 并行策略：每个 PID 处理一段 Blocks (1D Grid)
    pid = tl.program_id(0)
    
    # 假设 Grid 维度映射逻辑已处理 (Batch, Head, Block_Chunk)
    # 这里简化演示核心逻辑
    block_start_idx = pid * BLOCK_SIZE_SCATTER
    
    offs_n = tl.arange(0, BLOCK_SIZE_SCATTER) + block_start_idx
    mask_check = offs_n < N_BLOCKS
    
    # 1. 向量化读取 Mask 和 Offset
    # 计算指针...
    ptr_mask = mask_ptr + ... 
    ptr_offs = offsets_ptr + ...
    
    vals_mask = tl.load(ptr_mask, mask=mask_check, other=0)
    vals_offs = tl.load(ptr_offs, mask=mask_check, other=0)
    
    # 2. 计算写入位置
    # 因为 cumsum 是 inclusive 的，索引 = val - 1
    write_loc = vals_offs - 1
    
    # 3. 写入 (Scatter)
    # 只有 mask=1 的线程才写入，避免无效内存访问
    # 目标地址基于 write_loc 计算
    target_ptr = indices_ptr + ... + write_loc * stride_idx_k
    
    # 只有需要保留的 block 才执行 store，且地址互不冲突
    tl.store(target_ptr, offs_n, mask=(mask_check & (vals_mask > 0)))
3. 其他辅助优化建议 (Secondary Optimizations)
针对 A800 硬件特性，建议在实施上述方案后，进一步微调以下参数以“隐藏延迟”：

增加流水线级数 (num_stages):

建议设置 num_stages=3 或 4。这允许 GPU 在计算当前数据的同时，预取下一批 Q/K/V 数据，对 Memory Bound 场景至关重要。

增加 Warp 数量 (num_warps):

尝试将 num_warps 从 4 提升至 8。这能增加 Scheduler 的候补 Warp 队列，当一个 Warp 等待内存时，可以立即切换到另一个 Warp 计算，减少 Stall。

内存合并 (Coalescing):

检查 K 的读取逻辑。确保 offs_n (对应 BLOCK_SIZE_K 维度) 是内存中连续的。对于量化数据 (int8)，Triton 需要以 16字节 (128 bit) 对齐读取效率最高。
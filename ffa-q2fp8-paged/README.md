# FFA Q2FP8 分页 KV Cache

本目录实现了分页 KV cache 的 decode 注意力实验：K 采用 q2 对称量化并打包，
V 保持全精度（由 CLI 的 `--dtype` 控制）。核心思路：
- 将 KV 以固定大小的页（page）存入全局缓存，支撑超长序列。
- K 量化为 2-bit 并以 4 个值打包到 1 字节，降低带宽压力。
- 预先计算每页的统计元信息，便于分析或后续剪枝策略。
- 采用“阈值估计 + 两阶段 paged attention”来跳过影响较小的页。

主要文件：
- `attn_kernel/attn_q2fp8_paged.py`：分页缓存结构、页面写入与元信息计算。
- `attn_kernel/attn_q2fp8_paged_attn.py`：阈值估计 + 两阶段 paged attention 内核。
- `run_attn_bench_q2fp8_paged_attn.py`：端到端 decode 基准与绘图。

## 数据布局与缓存结构

本目录推荐使用 head-major 布局。

Paged KV cache：
- `k_q`（q2 K 打包）：`[Npage, HKV, SBS, K_PACKED]`，dtype `uint8`。
- `v`（全精度 V）：`[Npage, HKV, SBS, V]`，dtype 由 `--dtype` 指定。
- `k_scale`：`[B, HKV, K]` 或 `[HKV, K]`（广播到 B=1）。
- `page_lens`：`[Npage]`，每页有效 token 数（最后一页可能更短）。

Block table（每个序列）：
- `block_table`：`[B, max_pages]`，序列内页索引 -> 全局 `page_id`。
- `page_counts`：`[B]`，每个序列有效页数。

每页元信息（每页/每头）：
- `k_page_absmax`：该页 K 反量化后的最大绝对值。
- `k_page_sumabs`：该页 K 反量化后的绝对值总和。
- `k_page_l2`：该页 K 反量化后的 L2 范数。

术语：
- `SBS`：page 大小（每页 token 数）。
- `HQ`：query head 数。
- `HKV`：key/value head 数。
- `G = HQ / HKV`：GQA 分组比例。
- `K`：Q/K 的 head 维度，`V`：V 的 head 维度。

## Q2 对称量化与打包

量化发生在 `run_attn_bench_q2fp8_paged_attn.py`
（见 `quantize_k_2bit_symmetric_packed`）：
1) 计算时间维上的 absmax（按 `(B, HKV, K)`）：
   - `k_absmax = max_t |k|`
2) 用 2-bit 对称零点计算 scale：
   - `qmax = 3`, `qzero = 1.5`
   - `scale = clamp(k_absmax / qzero, min=1e-6)`
3) 量化：
   - `k_q = clamp(round(k / scale + qzero), 0, qmax)`
4) 4 个 2-bit 打包到 1 字节：
   - `k_q_packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)`

注意力计算中不会显式反量化 K，而是缩放 Q：
- `q_scaled = q * scale`
- `q_zero_sum = -qzero * sum(q_scaled)`
- `dot(q_scaled, k_q) + q_zero_sum` 等价于 `dot(q, dequantize(k_q))`
这样避免逐 token 反量化，并保持 K 在打包形式中读取。

## 页面写入与元信息计算

`attn_kernel/attn_q2fp8_paged.py` 提供：
- `allocate_paged_kv_cache`：分配全局页缓存与元信息缓冲区。
- `update_block_table`：向序列的 block table 追加 page id。
- `update_pages`：写入 K/V 页并可选计算元信息。
- `compute_pages_meta`：启动 `compute_page_meta_q2_packed` 计算页元信息。

`compute_page_meta_q2_packed`（Triton）：
- 遍历页内 token 与 K 维（以 `BK` 分块）。
- 解包 q2 值并乘 `k_scale`，计算：
  - absmax、sumabs、L2（每页/每头）。
- 用 `page_lens` 跳过最后页的 padding。
- 写入 `k_page_absmax`、`k_page_sumabs`、`k_page_l2`。

元信息在页面写入/更新时计算，因此后续内核可直接使用。
当前 attention 路径尚未使用这些元信息。

## 分页注意力解码算法

`PagedAttnRunner` 中的 paged attention 逻辑分三步（目前只支持 `B=1`）：

步骤 1：阈值估计（页剪枝启发式）
- 内核：`paged_attn_compute_threshold_qbits[_contig]`
- 仅使用“第一页”和“最后一页”计算每个 query head 的最大 logit。
- 阈值：`th = max(m_first, m_last) - delta`
- 若页是连续的（`page_id == page_index`），使用 contig 快路径。

步骤 2：逐页 softmax 与部分输出
- 内核：`paged_attn_stage1_qbits[_contig]`
- 对每页：
  - 计算该页所有 token 的 logit（q2 打包 K）。
  - `m_rows_blk = max(logits)`（按行）。
  - 若所有行都低于阈值，则剪枝该页。
  - 否则计算：
    - `b_p = exp2(logits - m_rows_blk)`
    - `l_rows = sum(b_p)`
    - `o_tile = b_p @ V`
  - 写入 `m_rows_blk`、`l_rows`、`o_tile` 和 keep mask。

步骤 3：跨页归并（带 mask）
- 内核：`paged_attn_stage2_masked`
- 只合并 keep 的页，使用 log-sum-exp 风格：
  - `new_m = max(m_prev, m_blk)`
  - `acc = acc * exp2(m_prev - new_m) + l_blk * exp2(m_blk - new_m)`
  - `o = o * exp2(m_prev - new_m) + o_blk * exp2(m_blk - new_m)`
- 最终输出：`o / acc`（按 query head）。

说明：
- logit 先乘 `1 / sqrt(K)`，再乘 `1 / ln(2)` 以便用 `exp2`。
- `delta` 控制剪枝激进程度（越大越少剪枝）。
- 阈值估计只看首尾页，属于近似启发式。

## Benchmark 流程

`run_attn_bench_q2fp8_paged_attn.py` 的步骤：
1) 从保存的层数据加载 Q/K/V。
2) 取 decode 查询（`q = q[:, :, L-1, :]`）。
3) K 做 q2 量化与打包，V 保持全精度。
4) 组装固定大小的页并写入全局 paged cache。
5) 运行 paged attention，可选与 FlashAttention 对比。
6) 保存原始结果并绘制速度曲线。

示例：
```bash
python run_attn_bench_q2fp8_paged_attn.py --SBS 256 --layer 1 --step 4096 --iters 200 --warmup 50 --delta 5.0
```

## 限制与假设
- `PagedAttnRunner` 目前只支持 `B=1`。
- 仅覆盖 decode 注意力（每个 head 单 query）。
- K 为 q2 打包，V 保持 `--dtype`（此目录暂无 fp8 kernel）。
- `k_scale` 为 `(B, HKV, K)` 的共享尺度，沿 token 维复用。
- block table 可非连续，但连续时会走 contig 快路径。

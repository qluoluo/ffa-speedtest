# Q2FP8 解码加速（ffa-q2fp8-threshold）

## 概述
本目录实现了自定义的解码期注意力路径：K 以 2-bit 量化存储并配合 FP8 残差，在阈值剪枝的基础上减少计算量。目录中还包含融合的 K 投影 + RoPE + 量化路径以及测速工具。

## 核心数据布局
- q: [B, 1, HQ, K]
- k_q: [B, T, HKV, K_packed]，其中 K_packed = ceil(K / 4)，每个字节打包 4 个 2-bit 值
- k_scale, k_zero: [B, HKV, K]（按 head、按通道缩放和零点，沿 T 共享）
- k_residual: [B, T, HKV, K]（用于反量化修正的 FP8 残差）
- v: [B, T, HKV, V]

## 关键组件
### 1) 融合 K 投影 + RoPE + int2/FP8 量化
文件：`fused_kproj_rope_q2fp8.py`
- 生成 RoPE cache 并对 K 应用 RoPE（逐 token）。
- 量化流程：
  - 在序列维 T 上计算每个 (B, HKV, K) 的 min/max。
  - scale = (max - min) / 3，zero = min。
  - 2-bit 量化（取值 0..3，四舍五入 ties-to-even）。
  - 每字节打包 4 个量化值。
  - residual = K_fp32 - dequant(K_q)，保存为 FP8（优先 float8_e5m2，否则 fp16）。
- Triton kernels：
  - `_kproj_rope_minmax_kernel`：计算投影+RoPE 后的 min/max。
  - `_kproj_rope_quant_kernel`：重算 K、应用 RoPE、量化并写入 int2 与 residual。
- 含参考实现与 smoke test，用于验证融合路径输出。
- 约束：head_dim 必须为偶数；block_pair 必须为偶数；仅支持 k_bits=2。

### 2) 量化解码注意力 + 阈值剪枝
文件：`attn_kernel/attn_kernel_v1210_fused_bsz_q2fp8.py`
- 阈值预计算 kernel（`attn_compute_threshold_qbits`）：
  - 每个 (B, HKV) 基于首块/末块计算阈值（按 head 输出）。
  - Stage 1 复用预计算阈值，避免每个 worker 重复计算。
- Stage 1 kernel（`attn_forward_stage1_fused_threshold_qbits`）：
  - 从打包 int2 + k_scale/k_zero 反量化 K。
  - 计算每个 block 的 max score，并根据阈值剪枝。
    - 默认由 `attn_compute_threshold_qbits` 预计算阈值。
    - 若 `use_ext_th=False`，可在 kernel 内用首块/末块估计阈值并减去 `delta`。
  - 若 head 组内所有 head 都低于阈值，则跳过该 block。
  - 可选残差修正（`USE_FP8_RESIDUAL`）在点积前加上 FP8 residual。
  - 写入每个 block 的 m/l/o 与 keep mask。
- Stage 2 kernel（`attn_forward_stage2_masked`）：
  - 对保留的 blocks 做 log-sum-exp 累加，得到最终输出。
- Python 包装 `attn_forward_decode_quantized` 校验 shape/dtype，并暴露：
  - `BS`（block size）、`SBS`（sub-block size）、`delta`（阈值余量）、
    `precomputed_threshold`（可选）和 `use_fp8_residual`。
- 输出 [B, HQ, V]，可选返回 skip ratio。

### 3) 基准与绘图
文件：`run_attn_bench_q2.py`
- 读取 layer 录制数据，转换布局，量化 K 为 int2 + FP8 residual，
  并与 FlashAttention 延迟对比。
- 结果缓存到 `plot/`，并绘制速度/跳过率曲线。

### 4) 工具
文件：`utils/*.py`
- `bench.py`：CUDA 计时工具。
- `load.py`：加载保存的 Q/K/V/hidden states，并支持截断。
- `cache.py`：基准结果 JSON 缓存读写。
- `plot.py`：绘制速度与跳过率曲线。
- `flash.py`：调用 FlashAttention 作为 baseline。

## 方法概览
该方法将 K 压缩为 2-bit 值并配合 FP8 残差以降低带宽消耗，同时用基于阈值的 block 剪枝策略跳过低分 block。量化 scale 为按 head、按通道设置，并在序列维 T 上共享，残差用于恢复部分精度。

# Q2FP8 Decode 加速清单

## 已实现
- **K_PACKED 一次性加载 + 四路 dot**：每个 token 只加载一次 packed 字节并通过位移解包，然后用 4 次小 dot 累加，避免每个 packed byte 被重复加载 4 次。
- **对齐提示（tl.multiple_of）**：对 k_q/k_res 的 packed token 基址加对齐提示，帮助编译器生成更好的合并加载。
- **向量化加载（新内核文件）**：在 `attn_kernel_v1210_fused_bsz_q2fp8_vec.py` 中加入 `tl.max_contiguous`/`tl.multiple_of` 的向量化加载提示，不影响原内核。

## 待优化
- **scale/zero 降精度**：在精度允许时把 `k_scale/k_zero` 存为 fp16/bf16，减少带宽占用。
- **融合阈值**：在 stage1 内部计算阈值，减少一次 kernel 启动开销。
- **Stage2 归约并行化**：把 `NTBS` 的串行循环改成分块/两级归约，降低长序列的尾部开销。
- **Autotune**：针对 H100 搜索 `num_warps/num_stages/BM_DOT/T_BS` 的最优组合。

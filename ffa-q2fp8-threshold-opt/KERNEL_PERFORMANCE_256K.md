# 256k Kernel 表现记录（按 GPU）

## 说明
- 数据来源：`plot/**/raw/*.json`。
- 仅统计长度为 262144（256k）的点。
- 指标：`q2_cg_ms`、`flash_ms`；相对基线为 `q2_cg_ms / baseline`。
- 基线：`attn_q2fp8_base_mask`（相同 GPU 与 meta 配置）。
- `N/A` 表示该项没有对应数据。
- 其余 kernel 与结果已移至 `backup/20250108_other_kernels/`。
- 默认对比配置：BS=128, SBS=128, bsz=1, delta=5.0, step=4096（除非表中另标注）。

## Kernel 版本差异（代码层面）
| kernel | 量化方式 | K 维度处理 | keep 列表 | Stage2 遍历 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `attn_q2fp8_base_mask` | 非对称（scale+zero-point） | K_PACKED 展开 | mask_buf | 扫描全部 NTBS | baseline |
| `attn_q2fp8_lr64_mask` | 非对称 | BK=64 分块 | mask_buf | 扫描全部 NTBS | 低寄存器路径 |
| `attn_q2fp8_sym_mask` | 对称（仅 scale） | K_PACKED 展开 | mask_buf | 扫描全部 NTBS | 去掉 zero-point |
| `attn_q2fp8_base_compact` | 非对称 | K_PACKED 展开 | kept_indices/kept_counts | 仅遍历 kept | 紧凑列表 |
| `attn_q2fp8_lr64_compact` | 非对称 | BK=64 分块 | kept_indices/kept_counts | 仅遍历 kept | 低寄存器 + 紧凑列表 |
| `attn_q2fp8_sym_lr64_compact` | 对称（仅 scale） | BK=64 分块 | kept_indices/kept_counts | 仅遍历 kept | 对称量化 + 低寄存器 + 紧凑列表 |

备注：compact 版本默认 `MAX_KEPT = ceil(0.2 * NTBS)`，且至少 32。

## NVIDIA-GeForce-RTX-4090_48GB

### 结果
| kernel | BS/SBS | bsz | layers | q2_cg_ms@256k | flash_ms@256k | 相对基线 | 相对flash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 1 | 1 | 0.256547 | 1.115281 | 1.000 | 0.230 |
| `attn_kernel/attn_q2fp8_lr64_mask.py` | 128/128 | 1 | 1 | 0.242016 | 1.114302 | 0.943 | 0.217 |
| `attn_kernel/attn_q2fp8_sym_mask.py` | 128/128 | 1 | 1 | 0.221540 | 1.114411 | 0.864 | 0.199 |
| `attn_kernel/attn_q2fp8_base_compact.py` | 128/128 | 1 | 1 | 0.225438 | 1.114405 | 0.879 | 0.202 |
| `attn_kernel/attn_q2fp8_lr64_compact.py` | 128/128 | 1 | 1 | 0.216936 | 1.115757 | 0.846 | 0.194 |
| `attn_kernel/attn_q2fp8_sym_lr64_compact.py` | 128/128 | 1 | 1 | 0.191838 | 1.114577 | 0.748 | 0.172 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 2 | 1-2 | 0.527991 | 2.207033 | 1.000 | 0.239 |

注：bsz=2 行为历史基线数据，未在本次重新跑。

### 性能总结
- 最快：`attn_q2fp8_sym_lr64_compact`（0.191838 ms，0.748x），较基线约快 25%。
- 紧凑列表带来的收益大于低寄存器分块：`base_compact` 0.879x vs `lr64_mask` 0.943x。
- 对称量化本身收益明显：`sym_mask` 0.864x；叠加低寄存器 + 紧凑列表后进一步提升到 0.748x。

### 方法加速比（相对 `attn_q2fp8_base_mask`）
| 方法 | 对应 kernel | q2_cg_ms@256k | 相对基线 | 加速比 |
| --- | --- | --- | --- | --- |
| 低寄存器 BK=64 | `attn_q2fp8_lr64_mask` | 0.242016 | 0.943 | 1.060x |
| 紧凑 keep 列表 | `attn_q2fp8_base_compact` | 0.225438 | 0.879 | 1.138x |
| 对称量化 | `attn_q2fp8_sym_mask` | 0.221540 | 0.864 | 1.158x |
| 低寄存器 + 紧凑列表 | `attn_q2fp8_lr64_compact` | 0.216936 | 0.846 | 1.183x |
| 对称量化 + 低寄存器 + 紧凑列表 | `attn_q2fp8_sym_lr64_compact` | 0.191838 | 0.748 | 1.337x |

## NVIDIA-H100-80GB-HBM3_80GB

### 结果
| kernel | BS/SBS | bsz | layers | q2_cg_ms@256k | flash_ms@256k | 相对基线 | 相对flash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 1 | 1 | 0.259155 | 0.355217 | 1.000 | 0.730 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 256/256 | 1 | 1 | 0.240535 | 0.355023 | 1.000 | 0.678 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 256/256 | 4 | 1-2-3-4 | 0.957358 | 1.348051 | 1.000 | 0.710 |
| `attn_kernel/attn_q2fp8_lr64_mask.py` | 128/128 | 1 | 1 | 0.295166 | 0.354962 | 1.139 | 0.832 |
| `attn_kernel/attn_q2fp8_sym_mask.py` | 128/128 | 1 | 1 | 0.231700 | 0.355080 | 0.894 | 0.653 |
| `attn_kernel/attn_q2fp8_base_compact.py` | 128/128 | 1 | 1 | 0.217692 | 0.355781 | 0.840 | 0.612 |
| `attn_kernel/attn_q2fp8_lr64_compact.py` | 128/128 | 1 | 1 | 0.209329 | 0.354954 | 0.808 | 0.590 |
| `attn_kernel/attn_q2fp8_sym_lr64_compact.py` | 128/128 | 1 | 1 | 0.231274 | 0.355477 | 0.892 | 0.651 |

### 性能总结
- 最快：`attn_q2fp8_lr64_compact`（0.209329 ms，0.808x）。
- `base_compact` 也有明显收益（0.840x）；`sym_mask` 与 `sym_lr64_compact` 约 0.89x。
- `lr64_mask` 在 H100 上反而变慢（1.139x），低寄存器分块不一定单独收益。

### 方法加速比（相对 `attn_q2fp8_base_mask`）
| 方法 | 对应 kernel | q2_cg_ms@256k | 相对基线 | 加速比 |
| --- | --- | --- | --- | --- |
| 低寄存器 BK=64 | `attn_q2fp8_lr64_mask` | 0.295166 | 1.139 | 0.878x |
| 紧凑 keep 列表 | `attn_q2fp8_base_compact` | 0.217692 | 0.840 | 1.190x |
| 对称量化 | `attn_q2fp8_sym_mask` | 0.231700 | 0.894 | 1.118x |
| 低寄存器 + 紧凑列表 | `attn_q2fp8_lr64_compact` | 0.209329 | 0.808 | 1.238x |
| 对称量化 + 低寄存器 + 紧凑列表 | `attn_q2fp8_sym_lr64_compact` | 0.231274 | 0.892 | 1.121x |

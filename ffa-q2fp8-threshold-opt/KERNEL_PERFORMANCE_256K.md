# 256k Kernel 表现记录（按 GPU）

## 说明
- 数据来源：`plot/**/raw/*.json`。
- 仅统计长度为 262144（256k）的点。
- 指标：`q2_cg_ms`、`flash_ms`；相对基线为 `q2_cg_ms / baseline`。
- 基线：`attn_q2fp8_base_mask`（相同 GPU 与 meta 配置）。
- `N/A` 表示该项没有对应数据。
- 其余 kernel 与结果已移至 `backup/20250108_other_kernels/`。

## NVIDIA-GeForce-RTX-4090_48GB

### 结果
| kernel | BS/SBS | bsz | layers | q2_cg_ms@256k | flash_ms@256k | 相对基线 | 相对flash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 1 | 1 | 0.254781 | 1.117606 | 1.000 | 0.228 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 2 | 1-2 | 0.527991 | 2.207033 | 1.000 | 0.239 |
| `attn_kernel/attn_q2fp8_sym_mask.py` | 128/128 | 1 | 1 | 0.221411 | 1.114454 | 0.869 | 0.199 |
| `attn_kernel/attn_q2fp8_lr64_compact.py` | 128/128 | 1 | 1 | 0.188983 | 1.120860 | 0.742 | 0.169 |

### 潜力结论
- 优先：`attn_q2fp8_lr64_compact`（0.742x）。
- 候选：`attn_q2fp8_sym_mask`（0.869x）。

## NVIDIA-H100-80GB-HBM3_80GB

### 结果
| kernel | BS/SBS | bsz | layers | q2_cg_ms@256k | flash_ms@256k | 相对基线 | 相对flash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attn_kernel/attn_q2fp8_base_mask.py` | 128/128 | 1 | 1 | 0.259155 | 0.355217 | 1.000 | 0.730 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 256/256 | 1 | 1 | 0.240535 | 0.355023 | 1.000 | 0.678 |
| `attn_kernel/attn_q2fp8_base_mask.py` | 256/256 | 4 | 1-2-3-4 | 0.957358 | 1.348051 | 1.000 | 0.710 |
| `attn_kernel/attn_q2fp8_sym_mask.py` | 128/128 | 1 | 1 | N/A | N/A | N/A | N/A |
| `attn_kernel/attn_q2fp8_lr64_compact.py` | 128/128 | 1 | 1 | 0.209329 | 0.354954 | 0.808 | 0.590 |

### 潜力结论
- 优先：`attn_q2fp8_lr64_compact`（0.808x）。
- 待补：`attn_q2fp8_sym_mask`（H100 256k 数据缺失）。

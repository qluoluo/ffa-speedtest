# FFA-Sample-Reorder

Sample4 + FP16 + Reorder 加速方案，基于 ffa-sample 的 attn_sample4_fp16 改进。

## 核心改进

原版 `attn_sample4_fp16`:
- 每个 Block128 采样 4 个点（位置 [0, 32, 64, 96]）
- 采样 K 额外存储为 `[B, num_blocks, HKV, NUM_SAMPLES, K]`
- 存储开销: `num_blocks * 4 * K * 2 bytes` (FP16)

Reorder 版本:
- 将采样点交换到序列维度的最前面
- 无需额外存储采样 K
- 直接从重排后的 K 读取采样点

## 数据布局

重排后的 K/V 布局:
```
k_reordered: [B, T, HKV, K]
  - 位置 [0, num_blocks*4): 所有采样点
    - block 0 的采样点: [0, 4)
    - block 1 的采样点: [4, 8)
    - ...
  - 位置 [num_blocks*4, T): 剩余 token
```

## 文件结构

```
ffa-sample-reorder/
├── attn_kernel/
│   ├── __init__.py
│   └── attn_sample4_fp16_reorder.py  # 核心 kernel
├── test_attn_sample4_fp16_reorder.py # 测试脚本
├── .gitignore
└── AGENTS.md
```

## 使用方法

```python
from attn_kernel.attn_sample4_fp16_reorder import (
    reorder_kv_for_sampling,
    attn_forward_decode_reorder,
    CUDAGraphDecodeRunnerReorder,
)

# 1. 重排 K/V（预处理阶段，只需执行一次）
k_reordered, v_reordered, reorder_indices, inverse_indices = reorder_kv_for_sampling(k, v, BS=128)

# 2. 执行 attention
out = attn_forward_decode_reorder(
    q=q,
    k_reordered=k_reordered,
    v_reordered=v_reordered,
    BS=128,
    delta=5.0,
)

# 3. 可选：使用 CUDAGraph 加速
runner = CUDAGraphDecodeRunnerReorder(q, k_reordered, v_reordered, BS=128, delta=5.0)
out = runner.replay_only()
```

## 测试

```bash
python test_attn_sample4_fp16_reorder.py
python test_attn_sample4_fp16_reorder.py --T 131072 --skip-correctness
```

## 内存节省示例

对于 T=65536, HKV=8, K=128, BS=128:
- 原版采样 K 存储: ~4 MB
- Reorder 版本: ~0 MB (仅索引用于调试)
- 节省: ~100%

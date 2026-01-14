# Sample4 + 2-bit 量化加速方案

## 核心思想

结合采样和量化两种方法来加速 decode attention：
1. **采样减少计算量**：每个 Block128 只用 4 个采样点进行快速筛选（计算量 4/128 = 1/32）
2. **量化减少存储带宽**：采样点使用 2-bit 对称量化（存储量 2/16 = 1/8）

## 算法流程

### 1. KV Cache 准备
- 完整 K cache: `[B, T, HKV, K]` - 用于非剪枝 block 的精确计算
- 完整 V cache: `[B, T, HKV, V]` - 用于输出计算
- 采样 K (量化): `[B, num_blocks, HKV, 4, K_packed]` - 用于快速筛选
- 采样 scale: `[B, num_blocks, HKV, K]` - 量化参数

### 2. 阈值计算
使用第一个和最后一个 block 的采样点计算阈值：
```
threshold = max(score_first_block, score_last_block) - delta
```

### 3. Stage1: 快速筛选 + 精确计算
对每个 block：
1. 用 4 个采样点的量化 K 计算近似分数
2. 如果 `max(采样分数) < threshold`，剪枝整个 block
3. 否则，用完整 K 精确计算 attention score 和 output

### 4. Stage2: 合并输出
使用 online softmax 合并所有非剪枝 block 的输出。

## 数据格式

### 采样位置
Block Size = 128，采样间隔 = 32，采样位置 = [0, 32, 64, 96]

### 2-bit 对称量化
```python
scale = max(|k_sample|) / 1.5  # per block, per head, per dim
k_q = round(k / scale + 1.5)   # clamp to [0, 3]
```

## 性能预期

根据分析结论（Block Size = 128）：
- **一致率**: 99.11%
- **误剪率**: 0.89%
- **筛选计算量**: 原来的 1/32
- **筛选存储量**: 原来的 1/8

## 文件结构

```
ffa-sample4-q2/
├── attn_kernel/
│   ├── __init__.py
│   └── attn_sample4_q2_sym.py    # 主内核实现
├── utils/                         # 工具函数（从 ffa-q2fp8-threshold-opt 复制）
│   ├── bench.py
│   ├── cache.py
│   ├── flash.py
│   ├── load.py
│   └── plot.py
├── run_attn_bench_sample4_q2.py  # 基准测试脚本
└── AGENTS.md                      # 本文档
```

## 使用方法

```bash
# 运行基准测试
python run_attn_bench_sample4_q2.py --layer 1 --BS 128 --delta 5.0

# 完整参数
python run_attn_bench_sample4_q2.py \
    --layer 1 \
    --bsz 1 \
    --BS 128 \
    --SBS 128 \
    --delta 5.0 \
    --step 4096 \
    --iters 500 \
    --warmup 100 \
    --with-baseline \
    --profile-kernels
```

## 与其他方法对比

| 方法 | 存储 | 计算量 | 一致率 |
|------|------|--------|--------|
| FP 全量 | 16 bit | 100% | 100% |
| Q2FP8 | 2+8 bit | 100% | 99.35% |
| Sample4 FP | 16 bit | 4/BS | 99.14% |
| **Sample4+Q2** | 2 bit | 4/BS | **99.11%** |

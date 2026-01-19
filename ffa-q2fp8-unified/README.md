# Q2FP8 Unified Kernel Benchmark

对比 Q2FP8 Unified Kernel 和 FlashAttention 的性能。

## 功能特点

- 测试 Q2FP8 Unified Kernel（统一处理量化 + FP16 current tokens）
- 对比 FlashAttention baseline
- 支持不同的 current tokens 数量（0-128）
- 自动生成性能对比图表

## 使用方法

### 基本用法

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/q2fp8-unified-bench

# 默认配置：64 个 current tokens
python run_unified_bench.py --layer 1 --delta 5.0 --current-len 64

# 不使用 current tokens
python run_unified_bench.py --layer 1 --delta 5.0 --current-len 0

# 使用 128 个 current tokens
python run_unified_bench.py --layer 1 --delta 5.0 --current-len 128
```

### 参数说明

- `--dtype`: 数据类型 (fp16/bf16/fp32)，默认 fp16
- `--BS`: 量化 block 大小，默认 128
- `--SBS`: Sub-block 大小，默认等于 BS
- `--delta`: Threshold delta，默认 5.0
- `--layer`: 起始层索引，默认 1
- `--bsz`: Batch size（合并多少层），默认 1
- `--current-len`: FP16 current tokens 数量（0-128），默认 64
- `--max-current`: Current buffer 最大大小，默认 128
- `--max-length`: 截断序列长度
- `--step`: 长度扫描步长，默认 1024
- `--iters`: Benchmark 迭代次数，默认 500
- `--warmup`: Warmup 次数，默认 100
- `--no-flash`: 跳过 FlashAttention baseline
- `--no-plot`: 跳过绘图

### 示例

```bash
# 测试不同 current_len 的性能
python run_unified_bench.py --layer 1 --current-len 0 --iters 500
python run_unified_bench.py --layer 1 --current-len 32 --iters 500
python run_unified_bench.py --layer 1 --current-len 64 --iters 500
python run_unified_bench.py --layer 1 --current-len 128 --iters 500

# 测试不同 delta 的影响
python run_unified_bench.py --layer 1 --delta 3.0 --current-len 64
python run_unified_bench.py --layer 1 --delta 5.0 --current-len 64
python run_unified_bench.py --layer 1 --delta 7.0 --current-len 64

# 快速测试（少量迭代）
python run_unified_bench.py --layer 1 --current-len 64 --iters 100 --warmup 20
```

## 输出

### 1. 性能数据

保存在 `plot/q2fp8_unified/{GPU}/delta{delta}_layers{layer}_BS{BS}_SBS{SBS}_bsz{bsz}_curr{current_len}/raw/` 目录下，JSON 格式。

### 2. 性能图表

保存在 `plot/q2fp8_unified/{GPU}/delta{delta}_layers{layer}_BS{BS}_SBS{SBS}_bsz{bsz}_curr{current_len}/` 目录下，PNG 格式。

图表包含：
- Q2FP8 Unified Kernel 延迟曲线
- FlashAttention 延迟曲线
- Skip ratio 曲线（右侧 Y 轴）

### 3. 终端输出

```
[Result] Layers 1 | bsz=1 | T=256k | curr=64 | BS=128 SBS=128 delta=5.0 |
         Unified=1.234 ms, Flash=5.678 ms, Speedup=4.60x
[Result] Saved plot to: plot/q2fp8_unified/.../layer_1_speed_Tmax256k_unified_curr64.png
```

## 算法说明

Q2FP8 Unified Kernel 的核心特点：

1. **统一处理**：将 FP16 current tokens 作为特殊 block 统一处理
2. **固定 buffer**：使用固定大小的 current buffer（128 tokens）
3. **对称量化**：2-bit 对称量化 + FP8 残差
4. **动态剪枝**：基于 threshold 的 block 剪枝

## 依赖

- PyTorch with CUDA
- Triton
- flash-attn (可选，用于 baseline)
- matplotlib (用于绘图)
- tqdm

## 目录结构

```
q2fp8-unified-bench/
├── run_unified_bench.py    # 主测速脚本
├── utils/
│   ├── bench.py            # Benchmark 工具
│   ├── cache.py            # 缓存工具
│   └── flash.py            # FlashAttention wrapper
├── plot/                   # 输出目录
│   └── q2fp8_unified/
│       └── {GPU}/
│           └── delta{delta}_layers{layer}_BS{BS}_SBS{SBS}_bsz{bsz}_curr{current_len}/
│               ├── raw/    # JSON 数据
│               └── *.png   # 图表
└── README.md
```

## 注意事项

1. 确保 unified kernel 路径正确：`../e2e/q2fp8-unified/attn_kernel/attn_q2fp8_unified.py`
2. 确保数据路径存在：`/inspire/hdd/project/.../layer_data/`
3. 首次运行会较慢（需要 JIT 编译 Triton kernels）
4. 结果会自动缓存，重复运行会直接读取缓存

## 性能预期

在 256K 序列 + 64 current tokens 的情况下：
- Q2FP8 Unified: ~1.5 ms
- FlashAttention: ~6-8 ms
- Speedup: ~4-6x

实际性能取决于：
- GPU 型号
- 序列长度
- Current tokens 数量
- Skip ratio（真实数据通常 >99%）

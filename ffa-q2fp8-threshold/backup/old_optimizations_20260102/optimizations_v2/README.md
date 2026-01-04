# Q2FP8 Attention Kernel Optimizations

本目录包含针对H100优化的Q2FP8 attention decode kernel的各项优化实现。

## 📁 目录结构

```
q2fp8_optimizations/
├── kernels/                    # 优化kernel实现
│   ├── opt2_adaptive_bm_dot.py    # 优化2: 自适应BM_DOT
│   ├── opt3_fp16_obuf.py          # 优化3: FP16 o_buf
│   ├── opt4_stage2_compact.py     # 优化4: Stage2紧凑化
│   └── opt5_autotune.py           # 优化5: Triton autotune
├── benchmarks/                 # Benchmark脚本
│   └── benchmark_all_opts.py      # 统一benchmark脚本
├── results/                    # Benchmark结果 (自动生成)
└── run_all_benchmarks.sh       # 一键运行所有benchmark
```

## 🎯 优化说明

### 优化2: 自适应 BM_DOT (Adaptive BM_DOT)
**优化点:** 根据 G (group size) 自动选择最优的 BM_DOT 大小
- G ≤ 4: BM_DOT = 4
- G ≤ 8: BM_DOT = 8
- G > 8: BM_DOT = 16

**预期收益:** 20-30% Stage1 性能提升 (对于 G=3 的配置)

**原理:** 当前kernel使用固定的 BM_DOT=16，但对于小的 G 值（如 G=3）会浪费大量 warps。自适应选择可以减少warp浪费和shared memory压力。

### 优化3: FP16 o_buf (FP16 Output Buffer)
**优化点:** Stage1 使用 FP16 存储 o_buf，Stage2 读取时转换为 FP32

**预期收益:** 减少 50% Stage1→Stage2 内存流量，5-10% 总时间

**原理:** 原实现中 o_buf 使用 FP32 存储，占用大量带宽。使用 FP16 存储可以减半带宽，同时在 Stage2 中用 FP32 累加保证精度。

### 优化4: Stage2 紧凑化 (Stage2 Compact Iteration)
**优化点:** Stage2 只遍历被保留的blocks，而不是所有NTBS blocks

**预期收益:** Stage2 时间减少 30-50% (但Stage2只占总时间的16%)

**原理:** 当前Stage2遍历所有blocks并检查mask，即使skip ratio很高。使用compact keep-list可以直接跳过被剪枝的blocks。

### 优化5: Triton Autotune
**优化点:** 自动调优 num_warps 和 num_stages 参数

**预期收益:** 10-20% 性能提升（取决于硬件）

**原理:** 不同的GPU架构和配置需要不同的warp/stage设置。Autotune可以自动找到最优配置。

## 🚀 快速开始

### 1. 安装依赖 (仅需运行一次)

确保你的环境有以下依赖:
```bash
pip install torch>=2.0.0 triton>=2.1.0 numpy
```

### 2. 复制优化文件到项目

```bash
# 在你的ffa-q2fp8-threshold项目根目录
cp -r /tmp/q2fp8_optimizations/* ./
```

### 3. 运行benchmark (在4090上测试)

```bash
# 一键运行所有benchmark
./run_all_benchmarks.sh
```

或者手动运行:

```bash
python3 benchmarks/benchmark_all_opts.py \
    --batch-size 1 \
    --seq-len 262144 \
    --BS 256 \
    --SBS 256 \
    --delta 5.0 \
    --warmup 10 \
    --iters 100 \
    --output results/benchmark_results.json
```

### 4. 查看结果

Benchmark完成后会生成:
- `results_<timestamp>/benchmark_results.json` - 详细结果数据
- `results_<timestamp>/benchmark_log.txt` - 完整日志

## 📊 结果分析

### 预期性能提升 (H100)

| 优化 | Stage1改进 | Stage2改进 | 总体加速比 |
|------|-----------|-----------|----------|
| Baseline | - | - | 1.00x |
| Opt2 (Adaptive BM_DOT) | 20-30% | - | 1.15-1.25x |
| Opt3 (FP16 o_buf) | 5-10% | - | 1.05-1.10x |
| Opt4 (Stage2 Compact) | - | 30-50% | 1.05-1.08x |
| Opt5 (Autotune) | 10-20% | - | 1.10-1.20x |

**注意:** 实际性能取决于具体配置和硬件。

### 在H100上运行

```bash
# 1. 将整个目录复制到H100机器
scp -r q2fp8_optimizations/ user@h100-machine:/path/to/ffa-q2fp8-threshold/

# 2. SSH到H100
ssh user@h100-machine

# 3. 进入目录并运行
cd /path/to/ffa-q2fp8-threshold
./run_all_benchmarks.sh
```

## 🔍 故障排查

### 问题: ImportError: cannot import optimized kernels

**解决方案:**
1. 确保kernel文件在正确位置: `kernels/opt*.py`
2. 检查Python路径是否正确
3. 尝试手动导入测试:
   ```python
   from kernels.opt2_adaptive_bm_dot import attn_forward_decode_quantized_opt2
   ```

### 问题: CUDA out of memory

**解决方案:**
1. 减小序列长度: `--seq-len 131072`
2. 减小 BS/SBS: `--BS 128 --SBS 128`
3. 使用更小的batch size

### 问题: Triton compilation errors

**解决方案:**
1. 更新Triton版本: `pip install --upgrade triton`
2. 清除Triton cache: `rm -rf ~/.triton/cache`

## 📝 自定义配置

### 修改benchmark参数

编辑 `benchmarks/benchmark_all_opts.py` 中的默认参数，或使用命令行参数:

```bash
python3 benchmarks/benchmark_all_opts.py \
    --batch-size 1 \
    --seq-len 131072 \    # 减小序列长度
    --BS 128 \            # 减小block size
    --SBS 128 \
    --delta 5.0 \
    --warmup 20 \         # 增加warmup
    --iters 200           # 增加迭代次数
```

### 集成到现有代码

要在你的代码中使用优化kernel:

```python
# 导入优化kernel
from kernels.opt5_autotune import attn_forward_decode_quantized_opt5

# 替换原有的kernel调用
output = attn_forward_decode_quantized_opt5(
    q, k_q, k_scale, k_zero, v,
    k_residual=k_residual,
    BS=256, SBS=256, delta=5.0,
    use_fp8_residual=True,
)
```

## 📈 性能分析建议

1. **先在4090上验证功能正确性**
   ```bash
   # 使用小配置快速测试
   python3 benchmarks/benchmark_all_opts.py --seq-len 4096 --iters 10
   ```

2. **在H100上进行完整benchmark**
   ```bash
   ./run_all_benchmarks.sh
   ```

3. **比较不同配置**
   ```bash
   # 测试不同BS/SBS组合
   for BS in 128 256 512; do
       for SBS in 128 256; do
           python3 benchmarks/benchmark_all_opts.py \
               --BS $BS --SBS $SBS \
               --output results/bs${BS}_sbs${SBS}.json
       done
   done
   ```

## ⚠️ 注意事项

1. **优化1 (Upper-bound pruning) 未包含**
   - 这是收益最大的优化，但需要修改K quantization流程
   - 需要预计算并缓存per-block K norm/absmax
   - 建议作为下一步单独实现

2. **CUDAGraph支持**
   - 当前优化kernels支持CUDAGraph
   - 参考原有的 `attn_kernel_v1210_fused_bsz_q2fp8_cudagraph.py` 包装方式

3. **数值精度**
   - Opt3使用FP16 o_buf可能在极端情况下影响精度
   - 建议先验证accuracy再部署到生产环境

## 🤝 贡献

如果你发现bug或有优化建议，欢迎提issue或PR!

## 📄 License

与原项目保持一致

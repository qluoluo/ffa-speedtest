# H100优化测试指南

## 📋 测试内容

本测试套件系统地测试以下优化方向（**不包含LUT/FP8/Async**，因为已证明无效）：

1. **Block Size优化** - 测试不同的BS和SBS组合
2. **Delta参数调优** - 测试不同的剪枝阈值
3. **Precomputed Threshold** - 测试预计算阈值的效果
4. **组合优化** - 测试各种优化的组合效果
5. **Scaling测试** - 在不同序列长度下验证效果

## 🚀 快速开始

### 在H100上运行完整测试

```bash
cd optimizations/benchmarks

# 运行完整测试套件（约10-15分钟）
./run_h100_optimization_tests.sh
```

测试完成后，所有结果会保存在 `h100_optimization_logs/` 目录中。

### 单独运行某个测试

```bash
# 测试Block Size优化
python test_h100_optimizations.py --test bs --T 65536

# 测试Delta参数
python test_h100_optimizations.py --test delta --T 65536

# 测试组合优化
python test_h100_optimizations.py --test combined --T 65536

# 运行所有测试
python test_h100_optimizations.py --test all --T 65536
```

## 📊 测试详情

### Test 1: Block Size优化

测试不同的Block Size配置：
- Original: BS=128, SBS=128
- BS=256, SBS=128
- BS=256, SBS=256
- BS=512, SBS=256
- BS=1024, SBS=256
- BS=512, SBS=512

**目标**：找到H100上最优的block size配置

**预期**：BS=512或1024时性能最好（利用H100的228KB shared memory）

### Test 2: Delta参数优化

测试不同的delta值（剪枝阈值）：
- delta = 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0

**目标**：找到性能和准确率的最佳平衡点

**预期**：delta越大，skip ratio越高，性能越好（但可能影响准确率）

### Test 3: Precomputed Threshold

对比：
- 每次都计算threshold（当前实现）
- 预计算threshold一次，多次复用（理论收益）

**目标**：量化threshold计算的开销

**预期**：如果能复用threshold，可节省10-15%时间

### Test 4: 组合优化

测试各种优化的组合：
- Original: BS=128, delta=5.0
- BS=512, delta=5.0
- BS=512, delta=6.0
- BS=512, delta=7.0
- BS=1024, delta=6.0

**目标**：找到最优的参数组合

**预期**：BS=512-1024 + delta=6-7 时效果最好

### Test 5: Scaling测试

在多个序列长度下测试最优配置：
- T = 16384, 32768, 65536, 131072, 262144

**目标**：验证优化在不同长度下的稳定性

**预期**：长序列上收益更明显

## 📁 输出文件说明

测试完成后，`h100_optimization_logs/` 目录包含：

```
h100_optimization_logs/
├── test1_block_size_TIMESTAMP.log       # Block Size测试日志
├── test1_block_size_TIMESTAMP.json      # Block Size测试数据（JSON）
├── test2_delta_TIMESTAMP.log            # Delta测试日志
├── test2_delta_TIMESTAMP.json           # Delta测试数据（JSON）
├── test3_threshold_TIMESTAMP.log        # Threshold测试日志
├── test3_threshold_TIMESTAMP.json       # Threshold测试数据（JSON）
├── test4_combined_TIMESTAMP.log         # 组合测试日志
├── test4_combined_TIMESTAMP.json        # 组合测试数据（JSON）
└── test5_scaling_T*_TIMESTAMP.log/json  # 不同长度的测试结果
```

## 🔍 如何解读结果

### 关键指标

1. **Time (ms)** - 延迟时间，越低越好
2. **Skip ratio (%)** - 剪枝比例，越高说明计算量越少
3. **Speedup vs FlashAttn** - 相对FlashAttention的加速比，越高越好
4. **Speedup vs Original** - 相对原始Q2FP8配置的加速比

### 示例输出解读

```
[Optimized BS + Delta]
  BS=512, SBS=256, delta=6.0
  Time: 0.1500 ms
  Skip ratio: 99.50%
  Speedup vs FlashAttn: 2.33x      <- 目标：达到2.5-3.5x
  Speedup vs Original Q2FP8: 1.52x <- 相对原始Q2FP8的提升
```

### 成功的标志

如果看到以下结果，说明优化成功：
- ✅ Speedup vs FlashAttn **> 2.0x** (理想：2.5-3.5x)
- ✅ Speedup vs Original Q2FP8 **> 1.3x**
- ✅ Skip ratio **> 99%**

## 📤 提交结果

测试完成后，分享以下文件：

```bash
# 压缩所有结果
cd optimizations/benchmarks
tar -czf h100_optimization_results.tar.gz h100_optimization_logs/

# 或者只分享最关键的log
cat h100_optimization_logs/test4_combined_*.log
```

## 🔧 自定义测试参数

```bash
# 自定义序列长度
python test_h100_optimizations.py --test combined --T 131072

# 调整迭代次数（更精确，但更慢）
python test_h100_optimizations.py --test all --warmup 50 --iters 200

# 保存到指定文件
python test_h100_optimizations.py --test combined --output my_results.json
```

## ⚠️ 注意事项

1. **GPU独占**：运行测试时确保GPU没有被其他进程占用
2. **内存要求**：长序列测试（T=262144）需要足够的GPU内存
3. **时间估算**：完整测试约需10-15分钟
4. **不包含LUT优化**：本测试专注于参数调优，不测试LUT/FP8/Async

## 📊 预期结果摘要

基于分析，预期的最优配置：

```python
最优配置：
- BS = 512 或 1024
- SBS = 256
- delta = 6.0 - 7.0

预期效果：
- 当前Q2FP8 (BS=128, delta=5.0): ~0.28ms @ 256k
- 优化后: ~0.15-0.18ms @ 256k
- Speedup vs FlashAttn: 2.0-2.5x (目标：2.5-3.5x)
```

## 🐛 故障排除

### 如果测试失败：

```bash
# 检查GPU状态
nvidia-smi

# 确认CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查依赖
python -c "from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized"
```

### 如果内存不足：

```bash
# 减小序列长度
python test_h100_optimizations.py --test combined --T 32768

# 或只测试小规模BS
python test_h100_optimizations.py --test bs --T 65536
```

## 📞 分析支持

测试完成后，将整个 `h100_optimization_logs/` 目录分享出来，我会帮你：
1. 分析哪个配置最优
2. 对比实际效果vs预期
3. 给出下一步的优化建议

---

**准备好了吗？** 运行 `./run_h100_optimization_tests.sh` 开始测试！

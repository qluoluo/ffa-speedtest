# H100测试快速指南

由于H100和本机共享存储，您可以直接在H100上访问这个目录运行测试。

## 在H100上运行测试

### 方法1：使用提供的脚本（推荐）

```bash
# SSH到H100
ssh h100-machine

# 进入共享目录
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold/optimizations_v2

# 运行H100测试脚本
./RUN_ON_H100.sh
```

### 方法2：手动运行

```bash
# SSH到H100
ssh h100-machine

# 进入目录
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold/optimizations_v2

# 直接运行benchmark
./run_all_benchmarks.sh
```

## 对比RTX 4090和H100结果

测试完成后，在任意一台机器上运行：

```bash
cd /inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold/optimizations_v2

# 自动对比最新的RTX 4090和H100结果
./compare_results.py
```

## 结果文件位置

由于是共享存储，所有结果会自动出现在同一个目录：

```
optimizations_v2/
├── results_RTX4090_20260102_190043/   # RTX 4090结果
│   ├── benchmark_results.json
│   └── benchmark_log.txt
└── results_H100_YYYYMMDD_HHMMSS/      # H100结果（运行后生成）
    ├── benchmark_results.json
    └── benchmark_log.txt
```

## 注意事项

1. 确保H100上的Python环境已安装必要的依赖：
   - PyTorch >= 2.0.0
   - Triton >= 2.1.0
   - numpy

2. 如果H100使用不同的conda/virtualenv环境，请先激活相应环境

3. 结果会自动包含GPU型号在目录名中，便于区分

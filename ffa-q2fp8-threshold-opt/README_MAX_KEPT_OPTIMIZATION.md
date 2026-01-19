# MAX_KEPT_RATIO 优化项目

## 快速开始

### 查看优化结果

```bash
# 查看完整报告
cat MAX_KEPT_OPTIMIZATION_REPORT.md

# 查看对比图表
# (使用你喜欢的图片查看器)
eog plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png
```

### 重新运行测试

```bash
# 快速测试 (推荐，约2分钟)
bash scripts/run_max_kept_quick.sh

# 完整测试 (约30分钟)
bash scripts/run_max_kept_sweep.sh

# 生成对比图表
python scripts/plot_max_kept_comparison.py
```

## 核心结论

**推荐配置：`max_kept_ratio = 0.02`**

### 性能提升

- ✅ 延迟降低 **7.3%** (0.196ms → 0.182ms)
- ✅ 内存减少 **90%** (410 → 41 slots)
- ✅ 加速比提升 (5.75x → 6.14x vs FlashAttention)

### 如何应用

修改 `attn_kernel/attn_q2fp8_sym_lr64_atomic_compact.py`:

```python
def attn_forward_decode_quantized(
    ...
    max_kept_ratio: float = 0.02,  # 从 0.2 改为 0.02
    ...
):
```

或在调用时指定：

```python
output = attn_forward_decode_quantized(
    q, k_q, k_scale, v,
    max_kept_ratio=0.02,  # 显式指定
    ...
)
```

## 文件说明

### 报告和文档
- `MAX_KEPT_OPTIMIZATION_REPORT.md` - 完整的优化分析报告
- `README_MAX_KEPT_OPTIMIZATION.md` - 本文件

### 可视化结果
- `plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/max_kept_ratio_comparison.png` - 性能对比图表

### 测速脚本
- `scripts/run_max_kept_sweep.sh` - 完整测速 (step=4096, 64个点)
- `scripts/run_max_kept_quick.sh` - 快速测速 (step=65536, 4个点)
- `scripts/plot_max_kept_comparison.py` - 生成对比图表

### 原始数据
- `plot/attn_q2fp8_sym_lr64_atomic_compact_cudagraph/.../raw/*.json` - 所有测速原始数据

## 实验配置

- **GPU**: NVIDIA GeForce RTX 4090 (48GB)
- **Kernel**: attn_q2fp8_sym_lr64_atomic_compact
- **参数**: BS=128, SBS=128, delta=5.0
- **序列长度**: 64K - 256K tokens

## 性能数据

### 256K tokens 性能对比

| max_kept_ratio | MAX_KEPT | 延迟 (ms) | vs Flash | 提升 |
|----------------|----------|-----------|----------|------|
| 0.20 (baseline)| 410      | 0.196     | 5.75x    | -    |
| 0.10           | 205      | 0.188     | 5.93x    | +4.0%|
| 0.05           | 103      | 0.184     | 6.06x    | +6.0%|
| **0.02**       | **41**   | **0.182** | **6.14x**| **+7.3%**|

## 后续优化方向

1. **动态 MAX_KEPT** (最优先)
   - 根据实际 skip_ratio 自适应调整
   - 预期额外提升 5-10%

2. **Stage2 向量化优化**
   - 优化 merge kernel 的串行循环
   - 使用 `tl.static_range` 展开

3. **自适应 BK 分块**
   - 短序列: BK=128
   - 长序列: BK=64

4. **Warp 参数调优**
   - Stage1: `--num-warps-s1 8 --num-stages-s1 3`
   - Stage2: `--num-warps-s2 4 --num-stages-s2 2`

## 问题排查

### 如果性能没有提升

1. 检查是否正确传递了 `max_kept_ratio` 参数
2. 确认使用的是 CUDAGraph replay 模式
3. 验证 skip_ratio 是否足够高 (>99%)

### 如果遇到 OOM

- 增大 `max_kept_ratio` (如 0.05 或 0.1)
- 检查是否有其他内存泄漏

## 联系方式

如有问题或建议，请查看：
- 完整报告: `MAX_KEPT_OPTIMIZATION_REPORT.md`
- 原始测速脚本: `scripts/run_attn_bench_q2fp8_cudagraph.py`

---

生成时间: 2026-01-19
实验环境: RTX 4090, CUDA 12.x, Triton

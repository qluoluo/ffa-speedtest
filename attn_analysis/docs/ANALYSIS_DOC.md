# Block筛选方法对比分析 - 详细文档

## 1. 背景与目标

在稀疏注意力机制中，我们希望跳过那些对最终注意力输出贡献很小的KV block，从而加速推理。本分析比较了四种block筛选方法：

1. **FP (Full Precision)**: 直接使用FP16/FP32计算注意力分数，作为ground truth
2. **INT2 量化筛选**: 使用2-bit量化的K计算近似分数
3. **MinMax 上界筛选**: 使用每个block的min/max值计算分数上界
4. **Norm 上界筛选**: 使用向量范数计算分数上界

## 2. 数据格式

### 输入数据
- **Q (Query)**: `[B, HQ, T, K]` - B=batch, HQ=query头数, T=序列长度, K=头维度
- **K (Key)**: `[B, HKV, T, K]` - HKV=KV头数 (GQA中 HQ = HKV * G)

### 分析场景
- 模拟decode阶段：取最后一个位置的query (`q = Q[:, :, -1, :]`)
- 将K按block_size分块：`K_blocks = K.reshape(B, HKV, num_blocks, block_size, K)`

## 3. 四种筛选方法详解

### 3.1 FP (Full Precision) - Ground Truth

**计算方式**：
```
score[head, block] = max_{t in block}(q · k_t) * scale
```
其中 `scale = 1 / sqrt(K_dim)`

**阈值计算**：
```
threshold[head] = max(score[head, 0], score[head, -1]) - delta
```
即取第一个block和最后一个block的分数最大值，再减去delta（默认5.0）

**剪枝判断**：
```
prune[head, block] = (score[head, block] < threshold[head])
```
如果block的最大分数低于阈值，则该block被剪掉

### 3.2 INT2 量化筛选

**量化过程** (对称量化):
1. 对每个block的每个维度，计算最大绝对值：`max_abs[d] = max(|k[:, d]|)`
2. 计算scale：`scale[d] = max_abs[d] / 1.5` (1.5是INT2的零点)
3. 量化：`k_q[d] = round(k[d] / scale[d] + 1.5)`, 值域 [0, 1, 2, 3]

**分数计算**：
```
# 预乘scale
q_scaled = q * scale

# 计算量化分数
score_raw = q_scaled · k_q

# 减去零点偏移
score = score_raw - 1.5 * sum(q_scaled)
score = score * attention_scale
```

**原理说明**：
- INT2量化用4个离散值(0,1,2,3)近似表示K的每个元素
- 量化误差较小，能保持分数的相对大小关系
- 计算开销比FP低，适合快速筛选

### 3.3 MinMax 上界筛选

**原理**：
对于点积 `q · k = sum_d(q[d] * k[d])`，我们可以计算其上界：
- 如果 `q[d] > 0`，则 `q[d] * k[d]` 最大值在 `k[d] = k_max[d]` 时取得
- 如果 `q[d] < 0`，则 `q[d] * k[d]` 最大值在 `k[d] = k_min[d]` 时取得

**计算方式**：
```python
# 对每个block计算每个维度的min和max
k_min = k_block.min(dim=time_axis)  # [K]
k_max = k_block.max(dim=time_axis)  # [K]

# 构造最优k
k_opt[d] = k_max[d] if q[d] > 0 else k_min[d]

# 计算上界
upper_bound = (q · k_opt) * scale
```

**剪枝判断**：
```
# 如果上界都低于阈值，说明真实分数一定低于阈值
prune = (upper_bound < threshold)
```

**特点**：
- 保守估计：如果上界 < 阈值，则真实分数一定 < 阈值（不会误剪）
- 可能有漏剪：上界 >= 阈值不代表真实分数 >= 阈值
- Block越大，上界越松（因为min/max差距越大）

### 3.4 Norm 上界筛选

**原理**：
利用 Cauchy-Schwarz 不等式：`q · k <= ||q|| * ||k||`

**计算方式**：
```python
# 计算block内K向量的最大范数
k_norm_max = max_{t in block}(||k_t||)

# 计算上界
upper_bound = ||q|| * k_norm_max * scale
```

**特点**：
- 上界非常松（因为向量方向信息丢失）
- 对block size不敏感
- 实际效果很差，几乎无法剪枝

## 4. 统计指标解释

### 4.1 剪枝率 (Prune Ratio)
```
prune_ratio = 被剪掉的block数 / 总block数
```
越高表示筛选效果越好（跳过越多计算）

### 4.2 决策一致率 (Agreement Rate) - 详细说明

**定义**: 衡量近似方法与FP Ground Truth在剪枝决策上的一致程度。

**计算公式**:
```python
# 对于每个 (head, block) 位置，比较两种方法的剪枝决策
fp_prune = fp_scores < threshold      # FP的剪枝决策 (True=剪, False=不剪)
method_prune = method_scores < threshold  # 近似方法的剪枝决策

# 一致率 = 决策相同的位置数 / 总位置数
agreement = (fp_prune == method_prune).float().mean()
```

**具体示例**:
```
假设有 HQ=24 个头, num_blocks=4096 个block
总共有 24 × 4096 = 98,304 个决策点

对于每个决策点 (h, b):
  - 如果 FP判断剪枝 且 方法也判断剪枝 → 一致 (True Positive)
  - 如果 FP判断保留 且 方法也判断保留 → 一致 (True Negative)
  - 如果 FP判断保留 但 方法判断剪枝 → 不一致 (False Positive / 误剪)
  - 如果 FP判断剪枝 但 方法判断保留 → 不一致 (False Negative / 漏剪)

一致率 = (TP + TN) / (TP + TN + FP + FN)
```

**数值范例**:
```
FP剪枝率: 99%  (即 99% 的block被FP判断为应该剪掉)
方法剪枝率: 98%

可能的情况:
  - 真正剪枝 (TP): 98% 的block被两者都判断为剪
  - 真正保留 (TN): 0.5% 的block被两者都判断为保留
  - 误剪 (FP): 0.5% 的block被方法判断为剪，但FP判断为保留
  - 漏剪 (FN): 1% 的block被方法判断为保留，但FP判断为剪

一致率 = 98% + 0.5% = 98.5%
误剪率 = 0.5%
漏剪率 = 1%
```

**注意事项**:
- 一致率是对称的：A与B的一致率 = B与A的一致率
- 误剪率和漏剪率是互补的：误剪率 + 漏剪率 = 1 - 一致率
- 当FP剪枝率很高时（如99%），即使一致率看起来不高（如95%），实际误剪率可能很低

### 4.3 误剪率 (False Positive Rate)
```
false_positive = (方法剪了 但 FP不剪) 的比例
```
**危险指标**：如果误剪重要的block，会导致输出错误

**计算公式**:
```python
false_positive = ((method_prune) & (~fp_prune)).float().mean()
# 等价于: 方法判断剪枝 AND FP判断保留 的比例
```

### 4.4 漏剪率 (False Negative Rate)
```
false_negative = (方法不剪 但 FP剪) 的比例
```
漏剪只是浪费计算，不会影响正确性

**计算公式**:
```python
false_negative = ((~method_prune) & (fp_prune)).float().mean()
# 等价于: 方法判断保留 AND FP判断剪枝 的比例
```

### 4.5 指标关系
```
一致率 = 1 - 误剪率 - 漏剪率

即: agreement + false_positive + false_negative = 1
```

### 4.6 分数误差 (Score Error)
```
error = |method_score - fp_score|
```
衡量近似分数与真实分数的差距

### 4.7 上界Gap
```
gap = upper_bound - fp_score
```
衡量上界的紧度，越小越好

## 5. 分析流程

```
1. 加载Q, K数据
   ↓
2. 将K分成blocks
   ↓
3. 对每个head、每个block计算:
   - FP分数 (ground truth)
   - INT2分数
   - MinMax上界
   - Norm上界
   ↓
4. 计算阈值 (使用第一个和最后一个block)
   ↓
5. 生成剪枝mask
   ↓
6. 计算统计指标:
   - 剪枝率
   - 方法间一致率
   - 误剪率/漏剪率
   - 分数误差
```

## 6. 文件结构

```
output_full_analysis/
├── blocksize_comparison.png     # 跨block size对比图
├── analysis_report.txt          # 详细文字报告
├── bs_8/
│   ├── results.pt               # 原始分析结果
│   └── full_analysis.png        # 详细分析图
├── bs_16/
│   └── ...
├── bs_32/
│   └── ...
└── ...
```

## 7. 关键发现预期

1. **INT2 vs FP**:
   - 高一致率 (>99%)
   - 极低误剪率 (<0.01%)
   - 分数误差小

2. **MinMax vs FP**:
   - 一致率随block size变化
   - 小block size效果好
   - 大block size几乎无效
   - 零误剪率（保守方法）

3. **Norm**:
   - 上界太松，无法剪枝

## 8. 使用建议

- **高精度场景**: 使用INT2量化筛选
- **快速预筛选**: 使用小block size (8-16) 的MinMax
- **两级筛选**: MinMax预筛选 + INT2精细筛选

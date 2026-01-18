# E2E Speed Test - Self-Contained Version

本目录包含了所有必要的模型和kernel代码,无需依赖外部路径。

## 目录结构

```
e2e/
├── q2fp8/                      # FFA-Q2FP8 测试
│   ├── ffa_model/              # FFA模型代码 (本地副本)
│   │   ├── modeling_llama.py   # Llama模型实现
│   │   ├── q2fp8_cache.py      # Q2FP8 Cache实现
│   │   ├── ffa_fwd_decode.py   # FFA decode接口
│   │   └── __init__.py
│   ├── attn_kernel/            # Attention kernel (本地副本)
│   │   ├── attn_q2fp8_sym_mask.py
│   │   ├── attn_q2fp8_sym_lr64_compact.py
│   │   └── ...
│   ├── run_e2e_test.py         # Q2FP8 独立测试脚本
│   └── compat_patch.py         # Transformers兼容补丁
│
├── quest/                      # Quest 测试
│   ├── quest_model/            # Quest模型代码 (本地副本)
│   │   ├── models/
│   │   │   ├── llama.py        # Quest Llama实现
│   │   │   └── QuestAttention.py
│   │   ├── ops/                # Quest operators
│   │   ├── utils/              # Quest utilities
│   │   └── __init__.py
│   └── run_e2e_test.py         # Quest 独立测试脚本
│
├── baseline/                   # Flash Attention 2 基准
│   └── run_e2e_test.py         # Baseline测试脚本
│
├── shared/                     # 共享工具
│   ├── benchmark_utils.py      # 基准测试工具
│   └── test_prompts.py         # 测试prompt
│
├── run_all.py                  # 对比测试脚本
├── test_imports.py             # 导入测试脚本
└── README.md                   # 本文件
```

## 使用方法

### 1. 测试导入是否正确

```bash
# 使用oc环境测试FFA-Q2FP8
conda activate oc
python test_imports.py

# 使用quest环境测试Quest
conda activate quest
python test_imports.py
```

### 2. 运行单独的测试

#### FFA-Q2FP8 测试
```bash
conda activate oc
cd q2fp8
python run_e2e_test.py --prompt_type medium --max_new_tokens 128
```

#### Quest 测试
```bash
conda activate quest
cd quest
python run_e2e_test.py --prompt_type medium --max_new_tokens 128
```

#### Flash Attention 2 基准测试
```bash
conda activate oc  # 或 quest
cd baseline
python run_e2e_test.py --prompt_type medium --max_new_tokens 128
```

### 3. 运行对比测试

```bash
# 比较所有方法 (需要在支持所有方法的环境中运行)
python run_all.py --prompt_type medium --max_new_tokens 128

# 只比较FFA和Baseline
python run_all.py --skip_quest --prompt_type medium --max_new_tokens 128

# 只比较Quest和Baseline
python run_all.py --skip_ffa --prompt_type medium --max_new_tokens 128
```

## 代码修改说明

所有外部路径引用已被替换为本地相对路径:

### q2fp8/run_e2e_test.py
```python
# 旧代码:
# ffa_source = Path("/inspire/.../ffa_q2fp8_sym")
# sys.path.insert(0, str(ffa_source))

# 新代码:
sys.path.insert(0, str(Path(__file__).parent / "ffa_model"))
```

### quest/run_e2e_test.py
```python
# 旧代码:
# quest_source = Path("/inspire/.../quest")
# sys.path.insert(0, str(quest_source))

# 新代码:
sys.path.insert(0, str(Path(__file__).parent / "quest_model"))
```

### q2fp8/ffa_model/ffa_fwd_decode.py
```python
# 旧代码:
# _KERNEL_PATH = "/inspire/.../attn_kernel"

# 新代码:
_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "..", "attn_kernel")
```

## 依赖环境

### oc 环境 (用于 FFA-Q2FP8 和 Baseline)
- transformers >= 4.45.2
- torch with CUDA
- flash-attn

### quest 环境 (用于 Quest)
- transformers >= 4.31.0
- torch with CUDA
- Quest kernels (已包含在 quest_model/ops/)

## 注意事项

1. **Quest kernel**: quest_model/ops/ 中包含编译好的 CUDA kernel (.so文件),确保与你的CUDA版本兼容
2. **CUDA Graph**: FFA-Q2FP8 支持 `--use_cudagraph` 参数,但目前在Python层面有限制
3. **内存**: 长序列测试可能需要大量GPU内存
4. **结果保存**: 测试结果会自动保存为JSON文件在各自的目录下

## 测试参数

常用参数:
- `--prompt_type`: short/medium/long/custom
- `--max_new_tokens`: 生成token数量
- `--num_runs`: 重复运行次数 (默认3)
- `--detailed`: 使用详细的token-by-token计时
- `--device`: 指定GPU设备 (默认cuda:0)

FFA-Q2FP8 特有参数:
- `--block_size`: 块大小 (默认128)
- `--k_bits`: 量化位数 (2或4)
- `--delta`: 阈值偏移 (默认5.0)
- `--use_cudagraph`: 启用CUDA Graph

Quest 特有参数:
- `--page_size`: 页大小 (默认16)
- `--token_budget`: token预算 (默认1024)

## 故障排除

### 导入错误
运行 `python test_imports.py` 检查所有导入是否正常

### Quest kernel错误
确保 quest_model/ops/ 中的 .so 文件存在且与CUDA版本匹配

### CUDA错误
检查GPU可用性: `python -c "import torch; print(torch.cuda.is_available())"`

### 路径错误
所有脚本都使用相对路径,确保从正确的目录运行

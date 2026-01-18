# 快速开始指南

## 已完成的工作

✅ 所有模型和kernel代码已复制到本地目录
✅ 所有外部路径引用已更新为相对路径
✅ 创建了测试脚本和文档

## 目录结构

```
e2e/
├── q2fp8/ffa_model/        # FFA-Q2FP8 模型 (本地副本)
├── q2fp8/attn_kernel/      # Attention kernels (本地副本)
├── quest/quest_model/      # Quest 模型 (本地副本)
├── baseline/               # Flash Attention 2 基准
├── shared/                 # 共享工具
├── test_imports.py         # 导入测试
└── README.md               # 完整文档
```

## 立即测试

### 1. 测试导入 (推荐先运行)

```bash
# 测试 FFA-Q2FP8
conda activate oc
python test_imports.py

# 测试 Quest
conda activate quest
python test_imports.py
```

### 2. 运行单个测试

```bash
# FFA-Q2FP8
conda activate oc
cd q2fp8
python run_e2e_test.py --prompt_type medium --max_new_tokens 128

# Quest
conda activate quest
cd quest
python run_e2e_test.py --prompt_type medium --max_new_tokens 128

# Baseline
conda activate oc
cd baseline
python run_e2e_test.py --prompt_type medium --max_new_tokens 128
```

### 3. 运行对比测试

```bash
# 比较 FFA vs Baseline
conda activate oc
python run_all.py --skip_quest --prompt_type medium --max_new_tokens 128

# 比较 Quest vs Baseline
conda activate quest
python run_all.py --skip_ffa --prompt_type medium --max_new_tokens 128
```

## 代码修改摘要

### 修改的文件:

1. **q2fp8/run_e2e_test.py**
   - 移除外部路径: `/inspire/.../ffa_q2fp8_sym`
   - 使用本地路径: `Path(__file__).parent / "ffa_model"`

2. **quest/run_e2e_test.py**
   - 移除外部路径: `/inspire/.../quest`
   - 使用本地路径: `Path(__file__).parent / "quest_model"`

3. **run_all.py**
   - 移除所有外部路径引用
   - 使用本地相对路径

4. **q2fp8/ffa_model/ffa_fwd_decode.py**
   - 移除硬编码kernel路径
   - 使用相对路径: `os.path.join(os.path.dirname(__file__), "..", "attn_kernel")`

### 复制的文件:

- **FFA-Q2FP8**: 
  - `modeling_llama.py`, `q2fp8_cache.py`, `ffa_fwd_decode.py` → `q2fp8/ffa_model/`
  - 所有 attention kernels → `q2fp8/attn_kernel/`

- **Quest**:
  - 完整的 quest 包 → `quest/quest_model/`
  - 包括 models/, ops/, utils/ 等

## 验证清单

- [x] FFA-Q2FP8 模型文件已复制
- [x] FFA-Q2FP8 kernel文件已复制
- [x] Quest 模型文件已复制
- [x] 所有路径引用已更新为相对路径
- [x] 创建了测试脚本 (test_imports.py)
- [x] 创建了文档 (README.md)

## 注意事项

1. **Quest kernel**: `quest_model/_kernels.cpython-310-x86_64-linux-gnu.so` 是符号链接,指向原始编译的kernel
2. **环境要求**: 
   - `oc` 环境用于 FFA-Q2FP8 和 Baseline
   - `quest` 环境用于 Quest
3. **GPU**: 所有测试需要 CUDA GPU

## 下一步

现在你可以:
1. 运行 `python test_imports.py` 验证所有导入
2. 运行单独的测试脚本
3. 运行对比测试
4. 查看 README.md 了解更多详细信息

所有代码现在都是自包含的,不再依赖外部路径! 🎉

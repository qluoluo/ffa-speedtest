# Q2FP8 CUDA Graph 快速开始

## 一分钟快速上手

### 1. 安装依赖

```bash
pip install torch transformers flash-attn triton
```

### 2. 运行示例

```bash
cd e2e/q2fp8-cudagraph

# 修改 example.py 中的 model_path
# model_path = "/path/to/your/llama/model"

python example.py
```

### 3. 运行测试

```bash
python test_cudagraph.py
```

## 核心代码示例

```python
import torch
from q2fp8_cudagraph_cache import Q2FP8CudaGraphCache
from modeling_llama import LlamaForCausalLM
from transformers import AutoConfig, AutoTokenizer

# 1. 配置
config = AutoConfig.from_pretrained(model_path)
config.attn_settings = {
    "use_ffa_decode": True,
    "use_cudagraph": True,
    "delta": 5.0,
    "BS": 128,
    "k_bits": 2,
}

# 2. 加载模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    device_map="cuda",
)

# 3. 创建 Cache
cache = Q2FP8CudaGraphCache(
    max_seq_len=4096,
    BS=128,
    k_bits=2,
)

# 4. 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    past_key_values=cache,
)
```

## 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_seq_len` | 2048-4096 | 根据应用场景选择 |
| `BS` | 128 | Block size,通常不需要修改 |
| `k_bits` | 2 | 2-bit 压缩比更高 |
| `delta` | 5.0 | 阈值偏移,越小速度越快 |

## 性能预期

在 H100 GPU 上:
- **加速比**: 1.5-2x (相比无 CUDA Graph)
- **吞吐量**: 140+ tokens/s (Llama-8B)
- **延迟**: ~7ms/token

## 常见问题

**Q: 显存不足怎么办?**
- 减小 `max_seq_len`
- 使用 `k_bits=2` 而不是 4

**Q: 如何禁用 CUDA Graph?**
- 设置 `use_cudagraph=False`

**Q: 支持哪些模型?**
- 目前支持 Llama 系列模型
- 其他模型需要修改 modeling 文件

## 下一步

- 阅读完整文档: [README.md](README.md)
- 查看测试代码: [test_cudagraph.py](test_cudagraph.py)
- 运行示例: [example.py](example.py)

## 技术支持

遇到问题?
1. 检查 CUDA 和 PyTorch 版本
2. 确认模型路径正确
3. 查看 README.md 中的常见问题
4. 提交 Issue

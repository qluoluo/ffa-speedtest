# ffa-speedtest

本仓库用于 FFA/量化注意力相关的测速与实验，按不同量化方案与硬件场景分目录组织。

## 目录结构

- `attn_analysis`：注意力分析脚本与结果整理。
- `ffa-fp8`/`ffa-fp4fp8`/`ffa-nf4fp8`/`ffa-nvfp4fp8`：不同量化组合的基准与工具。
- `ffa-fp8fp8`：fp8 + fp8 residual 相关 kernel、工具与 e2e 基准。
- `ffa-q2`/`ffa-q2fp8`/`ffa-q2fp8-h100`/`ffa-q2fp8-meanv`：q2 相关变体与基准。

## 使用说明

- 各目录内的脚本与依赖可能不同，请先阅读对应目录中的说明文件。
- 运行通常需要 CUDA 环境与相应依赖（例如 PyTorch、transformers、flash_attn、triton 等）。

### fp8fp8 e2e 基准（示例）

输入文本来自 `ffa-fp8fp8/e2e/input_text.txt`。基准脚本可进行 baseline 与 fp8fp8 解码对比，并在每个阶段清理显存。

```bash
python ffa-fp8fp8/e2e/bench_llama_fp8fp8.py --compare --mode decode
```

仅运行 fp8fp8 解码：

```bash
python ffa-fp8fp8/e2e/bench_llama_fp8fp8.py --ffa-decode --mode decode
```

## 备注

- 本仓库默认不自动运行任何测试或基准。
- 若显存紧张，可适当减小 `--seq-len` 或 `--batch`。

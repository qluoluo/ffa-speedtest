# 常用命令

说明：单点最长长度可用 `--step` 设为 `T_full`（如 262144），如长度不同请自行调整。

## H100 上基准对比（CUDAGraph replay-only，包含 FlashAttention baseline）
```bash
python run_attn_bench_q2_cudagraph.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 500 --warmup 100
```

## H100 上基准对比（非 CUDAGraph，包含 FlashAttention baseline）
```bash
python run_attn_bench_q2.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 100 --warmup 50
```

## 单点最长长度（CUDAGraph，step 设为 262144）
```bash
python run_attn_bench_q2_cudagraph.py --BS 256 --SBS 256 --delta 5 --step 262144 --iters 500 --warmup 100
```

## 单点最长长度（非 CUDAGraph，step 设为 262144）
```bash
python run_attn_bench_q2.py --BS 256 --SBS 256 --delta 5 --step 262144 --iters 100 --warmup 50
```

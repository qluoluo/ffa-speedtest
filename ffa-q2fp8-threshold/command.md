# 常用命令

## H100 上 CUDAGraph 分阶段计时（单点最长长度）
```bash
python run_attn_bench_q2_stage_timing_cudagraph.py --BS 256 --SBS 256 --delta 5 --length 262144 --iters 200 --warmup 50
```

## H100 上分阶段计时（非 CUDAGraph，单点最长长度）
```bash
python run_attn_bench_q2_stage_timing.py --BS 256 --SBS 256 --delta 5 --length 262144 --iters 200 --warmup 50
```

## 扫描全长度（CUDAGraph）
```bash
python run_attn_bench_q2_stage_timing_cudagraph.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 200 --warmup 50
```

## 扫描全长度（非 CUDAGraph）
```bash
python run_attn_bench_q2_stage_timing.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 200 --warmup 50
```

## 基准对比（CUDAGraph replay-only，包含 FlashAttention baseline）
```bash
python run_attn_bench_q2_cudagraph.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 500 --warmup 100
```

## 基准对比（非 CUDAGraph，包含 FlashAttention baseline）
```bash
python run_attn_bench_q2.py --BS 256 --SBS 256 --delta 5 --step 1024 --iters 100 --warmup 50
```

## 关闭 FP8 residual 的消融（单点最长长度）
```bash
python run_attn_bench_q2_stage_timing_cudagraph.py --BS 256 --SBS 256 --delta 5 --length 262144 --iters 200 --warmup 50 --no-fp8-residual
```

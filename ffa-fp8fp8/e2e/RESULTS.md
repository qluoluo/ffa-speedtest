# FP8FP8 e2e results

## Environment
- GPU: NVIDIA GeForce RTX 4090
- torch: 2.9.1+cu128
- batch=2, hidden=128, heads=4, kv_heads=2, layers=1, dtype=fp16
- iters=1, warmup=0

## Compare (baseline vs fp8fp8)

Commands:
```bash
python e2e/bench_llama_fp8fp8.py --mode decode --compare --seq-len 32768 --iters 1 --warmup 0
python e2e/bench_llama_fp8fp8.py --mode decode --compare --seq-len 65536 --iters 1 --warmup 0
```

Results:
```
seq_len  variant   ms/iter    ms/token  tok/s
32768    Baseline  45286.941  1.382     1447.1
32768    FP8FP8    46260.855  1.412     1416.7
65536    Baseline  91050.844  1.389     1439.5
65536    FP8FP8    90694.172  1.384     1445.2
```

## Compare (prefill + decode, decode-cudagraph)

Commands:
```bash
python e2e/bench_llama_fp8fp8.py --mode both --compare --seq-len 32768 --decode-tokens 256 --decode-cudagraph --greedy-decode --warmup 3 --iters 10
```

Config:
- batch=1, seq_len=32768, decode_tokens=256, dtype=fp16, fp8_dtype=e5m2
- hidden=3072, heads=24, kv_heads=8, layers=28
- decode_cudagraph=True, greedy=True, model_path=/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B

Results:
```
variant   prefill_ms/iter  prefill_tok/s  decode_ms/iter  ms/token  tok/s  total_ms/iter
Baseline  2879.358         11380.3        3307.197        12.919    77.4   6186.555
FP8FP8    2896.077         11314.6        2613.682        10.210    97.9   5509.759
```

Note: FP8FP8 decode skip ratio avg=0.9665 (samples=6912).
Note: token indices length warning during run (267552 > 131072).

## Decode + CUDA Graphs (static cache)

Commands:
```bash
python e2e/bench_llama_fp8fp8.py --mode decode --ffa-decode --seq-len 32768 --iters 1 --warmup 0 --decode-cudagraph
python e2e/bench_llama_fp8fp8.py --mode decode --ffa-decode --seq-len 65536 --iters 1 --warmup 0 --decode-cudagraph
```

Results:
```
seq_len  variant  ms/iter   ms/token  tok/s
32768    FP8FP8   3272.008  0.100     20029.3
```

Note: 64k decode-cudagraph triggered SIGSEGV in this environment, so no result.

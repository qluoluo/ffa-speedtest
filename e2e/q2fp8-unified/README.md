# FFA Q2FP8 Unified Implementation

This directory contains the unified Q2FP8 (2-bit quantized FP8) implementation for Fast Forward Attention (FFA).

## Directory Structure

```
.
├── attn_kernel/              # Attention kernel implementations
│   └── attn_q2fp8_unified.py # Unified Q2FP8 attention kernel
├── ffa_model/                # Core model implementations
│   ├── modeling_llama.py     # Modified LLaMA model with FFA support
│   ├── q2fp8_cache.py        # Q2FP8 KV cache implementation
│   ├── ffa_fwd_decode.py     # FFA forward decode logic
│   ├── fused_quantize_triton.py # Triton-based quantization kernels
│   ├── fused_rope_quant.py   # Fused RoPE and quantization
│   ├── cudagraph_wrapper.py  # CUDA Graph wrapper for optimization
│   ├── simple_cudagraph.py   # Simplified CUDA Graph implementation
│   └── oc_model.py           # Optimized computation model
├── archive/                  # Archived non-core files
│   ├── ffa_model_tests/      # Test files
│   ├── ffa_model_backup/     # Backup files
│   └── *.py, *.md            # Old benchmarks and documentation
└── benchmark_comparison.py   # Performance comparison script
```

## Core Components

### 1. Attention Kernel (`attn_kernel/`)
- **attn_q2fp8_unified.py**: Unified attention kernel with Q2FP8 quantization support

### 2. Model Implementation (`ffa_model/`)
- **modeling_llama.py**: LLaMA model with FFA decode support
- **q2fp8_cache.py**: Symmetric Q2FP8 KV cache with block-wise quantization
- **ffa_fwd_decode.py**: Fast forward decode implementation
- **fused_quantize_triton.py**: Triton kernels for efficient quantization
- **fused_rope_quant.py**: Fused RoPE (Rotary Position Embedding) and quantization
- **cudagraph_wrapper.py**: CUDA Graph optimization wrapper
- **simple_cudagraph.py**: Simplified CUDA Graph for decode optimization
- **oc_model.py**: Optimized computation utilities

## Benchmark Comparison

The `benchmark_comparison.py` script compares the performance of FFA-Q2FP8-Unified against Flash Attention baseline.

### Usage

```bash
python benchmark_comparison.py \
    --model_path /path/to/llama/model \
    --prefill_len 16384 \
    --decode_len 256 \
    --delta 5.0 \
    --BS 128 \
    --k_bits 2 \
    --device cuda \
    --dtype bfloat16 \
    --output_dir output
```

### Parameters

- `--model_path`: Path to the LLaMA model (default: Llama-3.1-8B)
- `--prefill_len`: Prefill sequence length (default: 16384)
- `--decode_len`: Number of decode tokens to generate (default: 256)
- `--delta`: Threshold delta for FFA attention selection (default: 5.0)
- `--BS`: Block size for quantization (default: 128)
- `--k_bits`: Quantization bits, 2 or 4 (default: 2)
- `--device`: Device to run on (default: cuda)
- `--dtype`: Model dtype: float16, bfloat16, or float32 (default: bfloat16)
- `--output_dir`: Output directory for results (default: output)
- `--align_to_bs`: Align cache to BS boundary before timed decode (default: True)
- `--warmup_decode_tokens`: Number of decode warmup tokens (default: 4)
- `--debug_stats`: Collect debug statistics

### Output

The script generates:
- JSON file with detailed benchmark results
- PNG plots comparing prefill/decode times and throughput
- Console output with speedup metrics

Results are saved to: `output/benchmark_comparison_{prefill_len}p_{decode_len}d_{timestamp}/`

## Key Features

1. **Q2FP8 Quantization**: 2-bit quantized FP8 KV cache for memory efficiency
2. **Block-wise Quantization**: Configurable block size (BS) for quantization granularity
3. **FFA Decode**: Fast forward attention for efficient decode phase
4. **CUDA Graph Optimization**: Reduced kernel launch overhead
5. **Fused Operations**: Combined RoPE and quantization for better performance

## Performance Metrics

The benchmark measures:
- **Prefill latency**: Time to process initial context
- **Decode latency**: Time to generate tokens
- **Decode throughput**: Tokens generated per second
- **Total time**: Combined prefill + decode time
- **Speedup**: FFA vs Flash Attention performance ratio

## Archive

The `archive/` directory contains:
- Old test files and benchmarks
- Documentation from previous iterations
- Backup implementations
- Experimental code

These files are kept for reference but are not part of the core implementation.

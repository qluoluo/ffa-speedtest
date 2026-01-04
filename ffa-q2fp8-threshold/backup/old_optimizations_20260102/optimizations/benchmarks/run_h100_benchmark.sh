#!/bin/bash
# H100专用benchmark脚本
# 运行完整的优化测试和性能对比

set -e

echo "=========================================="
echo "H100 Optimized Kernel Benchmark"
echo "=========================================="
echo ""

# 检查GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

if [[ ! "$GPU_NAME" =~ "H100" ]]; then
    echo "WARNING: This script is optimized for H100. Current GPU: $GPU_NAME"
    echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
    read
fi

echo ""
echo "=========================================="
echo "Step 1: Correctness Test"
echo "=========================================="
python test_optimized_kernels.py --test-correctness

echo ""
echo "=========================================="
echo "Step 2: Quick Benchmark (T=16384)"
echo "=========================================="
python test_optimized_kernels.py --bench --T 16384 --warmup 20 --iters 200

echo ""
echo "=========================================="
echo "Step 3: Long Sequence Benchmark (T=65536)"
echo "=========================================="
python test_optimized_kernels.py --bench --T 65536 --warmup 20 --iters 200

echo ""
echo "=========================================="
echo "Step 4: Very Long Sequence (T=131072)"
echo "=========================================="
python test_optimized_kernels.py --bench --T 131072 --warmup 20 --iters 100

echo ""
echo "=========================================="
echo "Step 5: Full Benchmark Suite"
echo "=========================================="
python test_optimized_kernels.py --full-bench --warmup 20 --iters 200

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "- Correctness tests: Check output above"
echo "- Performance data: See benchmark results above"
echo ""
echo "Next steps:"
echo "1. Compare speedup vs original kernel"
echo "2. Compare speedup vs FlashAttention"
echo "3. If speedup is good, integrate into your production code"
echo ""
echo "For integration, see README_OPTIMIZATIONS.md"

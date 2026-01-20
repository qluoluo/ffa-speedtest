# Quick Start Guide: Block-wise JIT CUDA Graph

## What Was Implemented

The `q2fp8-page` directory now contains a Block-wise JIT CUDA Graph implementation that accelerates Q2FP8 attention decoding by capturing and replaying CUDA kernels.

## Key Features

1. **Automatic Graph Capture**: Graphs are captured automatically when the number of full blocks changes
2. **Memory-Aware Invalidation**: Detects when `k_q` memory address changes (due to `torch.cat`)
3. **Fallback Support**: Automatically falls back to non-graph path when needed (LSE, skip_ratio)
4. **Zero Configuration**: Works out-of-the-box with existing Q2FP8 cache

## How It Works

### Graph Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Check: num_full_blocks > 0?                              │
│    └─ No  → Use regular attn_forward_decode                 │
│    └─ Yes → Continue to step 2                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Check: Need recapture?                                   │
│    - First run (current_graph_runner is None)               │
│    - Block count changed (num_full_blocks != cached)        │
│    - Memory address changed (k_q.data_ptr() != cached)      │
│    └─ No  → Skip to step 4 (replay existing graph)          │
│    └─ Yes → Continue to step 3                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Capture New Graph                                        │
│    - Free old graph runner                                  │
│    - Create CUDAGraphDecodeRunnerQ2FP8 (warmup=2)           │
│    - Update cached state (num_blocks, k_q_ptr)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Replay Graph                                             │
│    - Call current_graph_runner.replay(...)                  │
│    - Get output: [B, HQ, V]                                 │
│    - Reshape to [B, 1, HQ, V] for compatibility             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Check: Need LSE for merging?                             │
│    └─ Yes → Fallback to attn_forward_decode with return_lse │
│    └─ No  → Use graph output directly                       │
└─────────────────────────────────────────────────────────────┘
```

## Usage

No code changes required! The implementation is transparent to existing code:

```python
# Your existing code works as-is
model = LlamaForCausalLM.from_pretrained(...)
cache = Q2FP8SymCache(...)

# CUDA Graph acceleration happens automatically during decode
outputs = model.generate(
    input_ids,
    past_key_values=cache,
    max_new_tokens=100,
)
```

## Performance Expectations

- **First decode step**: Slightly slower (graph capture overhead)
- **Subsequent steps**: Faster (graph replay eliminates kernel launch overhead)
- **Block boundary**: Brief slowdown when recapturing (new block count)
- **With k_current**: Falls back to non-graph path (LSE needed for merging)

## Monitoring

To verify CUDA Graph is working, you can add debug prints:

```python
# In modeling_llama.py, line ~358
if need_recapture:
    print(f"[Layer {self.layer_idx}] Capturing graph: num_blocks={num_full_blocks}")
```

## Troubleshooting

### Graph not capturing?
- Check `cache_layer.num_full_blocks > 0`
- Verify `use_ffa_decode=True` in config

### Unexpected recaptures?
- Normal when `num_full_blocks` increases
- Check if `k_q` memory is being reallocated

### Performance not improving?
- CUDA Graph benefits are most visible with many decode steps
- Check if falling back to non-graph path (need_lse=True)

## Comparison: q2fp8 vs q2fp8-page

| Feature | q2fp8 (original) | q2fp8-page (new) |
|---------|------------------|------------------|
| CUDA Graph | ❌ No | ✅ Yes (Block-wise JIT) |
| Graph Invalidation | N/A | ✅ Block count + memory address |
| LSE Support | ✅ Yes | ✅ Yes (fallback path) |
| Memory Management | Standard | ✅ Aggressive (free old graphs) |
| Warmup Iterations | N/A | 2 |

## Next Steps

1. Run your existing benchmarks with `q2fp8-page`
2. Compare performance against `q2fp8` baseline
3. Monitor graph capture frequency
4. Adjust `warmup` parameter if needed (line 379)

## Files Modified

- `q2fp8-page/ffa_model/modeling_llama.py` - Main implementation
- `q2fp8-page/attn_kernel/attn_q2fp8_sym_mask.py` - Already contains CUDAGraphDecodeRunnerQ2FP8

## Support

For issues or questions, refer to:
- `IMPLEMENTATION_SUMMARY.md` - Detailed technical documentation
- Original `q2fp8/` directory - Reference implementation

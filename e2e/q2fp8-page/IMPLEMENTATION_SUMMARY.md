# Block-wise JIT CUDA Graph Implementation Summary

## Overview
This implementation adds Block-wise Just-In-Time (JIT) CUDA Graph acceleration to the Q2FP8 attention mechanism in the `q2fp8-page` directory.

## Key Changes

### 1. Import CUDAGraphDecodeRunnerQ2FP8
**File**: `q2fp8-page/ffa_model/modeling_llama.py`
**Location**: Lines 268-284

Added import statements to load the `CUDAGraphDecodeRunnerQ2FP8` class from the `attn_kernel.attn_q2fp8_sym_mask` module with fallback import paths.

### 2. State Variables in LlamaAttention.__init__
**File**: `q2fp8-page/ffa_model/modeling_llama.py`
**Location**: Lines 217-220

Added three state variables to track CUDA Graph lifecycle:
- `self.current_graph_runner = None` - Stores the active graph runner
- `self.cached_num_blocks = -1` - Tracks the block count of the current graph
- `self.cached_k_q_ptr = -1` - Tracks the memory address of k_q (invalidated by torch.cat)

### 3. Block-wise JIT CUDA Graph Logic
**File**: `q2fp8-page/ffa_model/modeling_llama.py`
**Location**: Lines 345-450

Implemented the core CUDA Graph logic with the following features:

#### Invalidation Check
The graph is recaptured when:
- `self.current_graph_runner is None` (first run)
- `num_full_blocks != self.cached_num_blocks` (block count changed)
- `k_q.data_ptr() != self.cached_k_q_ptr` (memory address changed due to torch.cat)

#### Graph Capture
When recapture is needed:
1. Free old graph runner (`self.current_graph_runner = None`)
2. Prepare kwargs excluding `return_skip_ratio` and `return_lse`
3. Set `use_fp8_residual=True` if config enables it
4. Instantiate `CUDAGraphDecodeRunnerQ2FP8` with `warmup=2`
5. Update cached state (`cached_num_blocks`, `cached_k_q_ptr`)

#### Graph Replay
- Call `self.current_graph_runner.replay(...)` with all necessary tensors
- Output shape: `[B, HQ, V]` (reshaped to `[B, 1, HQ, V]` for compatibility)

#### Fallback Paths
- **When `need_lse=True` or `return_skip=True`**: Falls back to non-graph `attn_forward_decode` since CUDA Graph doesn't support these features
- **When `num_full_blocks == 0`**: Uses regular path (no quantized blocks yet)

## Constraints Met

✅ **Only capture when `num_full_blocks > 0`**: Graph is only used when there are full quantized blocks

✅ **No LSE/residual updates in graph**: The graph path doesn't capture `return_lse` or skip_ratio computation

✅ **Aggressive memory management**: Old graph runners are explicitly freed before creating new ones

✅ **Maintains existing logic**: All other logic (merge_attention_output for k_current) remains outside the graph path

✅ **Post-processing shape compatibility**: Graph output `[B, HQ, V]` is reshaped to `[B, 1, HQ, V]` to match expected format

## Configuration

The CUDA Graph runner is configured with:
- `warmup=2` - Two warmup iterations before graph capture
- `use_fp8_residual=True` - Enabled if config supports it
- All other decode_kwargs passed through (except `return_skip_ratio` and `return_lse`)

## Benefits

1. **Reduced kernel launch overhead**: CUDA Graph captures and replays the entire kernel sequence
2. **Block-wise invalidation**: Only recaptures when block structure changes
3. **Memory-aware**: Tracks k_q pointer to detect torch.cat invalidations
4. **Backward compatible**: Falls back to non-graph path when needed (LSE, skip_ratio, no blocks)

## Testing Recommendations

1. Verify graph capture triggers correctly when `num_full_blocks` changes
2. Verify graph recapture when `k_q` memory address changes
3. Verify fallback to non-graph path when `need_lse=True`
4. Verify output shape compatibility with downstream processing
5. Monitor memory usage during graph capture/replay cycles

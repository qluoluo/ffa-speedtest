"""
CUDA Graph 集成补丁 - modeling_llama.py 修改说明

关键修改点：
1. 在 LlamaAttention 中添加 CUDA Graph runner 支持
2. 在 forward 方法中初始化和使用 CUDA Graph
3. 确保只在 buffer 初始化后才使用 CUDA Graph

使用方法：
    # 在 config 中启用 FFA decode
    config.attn_settings = {
        "use_ffa_decode": True,
        "delta": 5.0,
        "BS": 128,
        "SBS": 128,
        "use_fp8_residual": True,
        "k_bits": 2,
    }

    # 使用 Q2FP8SymCache
    from q2fp8_cache_optimized import Q2FP8SymCache
    cache = Q2FP8SymCache(BS=128, max_decode_tokens=4096)

    # Forward 会自动使用 CUDA Graph
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
"""

# ============================================================================
# 🆕 关键修改 1: 在 LlamaAttention.__init__ 中添加 CUDA Graph 支持
# ============================================================================
# 位置：class LlamaAttention(nn.Module): def __init__(...) 约在 Line 194

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # ... 原有代码 ...

        # 🆕 CUDA Graph 支持
        self.cudagraph_runner = None
        self.cudagraph_buffer_shape = None  # 用于检测 buffer shape 变化
        self.cudagraph_initialized = False


# ============================================================================
# 🆕 关键修改 2: 在 forward 方法中集成 CUDA Graph
# ============================================================================
# 位置：class LlamaAttention: def forward(...) 约在 Line 218
# 具体位置：在 decode 路径中（q_len == 1）

def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    # ... 原有代码（Q/K/V projection, RoPE 等）...

    # 检查是否使用 Q2FP8SymCache
    is_q2fp8_cache = isinstance(past_key_values, (Q2FP8SymCache, Q2FP8SymStaticCache))

    if is_q2fp8_cache:
        # ... 原有的 cache 更新代码 ...

        if q_len == 1:
            # ========== DECODE 路径 ==========
            # ... 原有的检查代码 ...

            # 获取 cache layer
            cache_layer = past_key_values.layers[self.layer_idx]

            # 🆕 检查 buffer 是否已初始化
            if not cache_layer.buffer_initialized:
                # Prefill 刚完成，buffer 还未初始化
                # 使用标准路径（只有第一次 decode 会走这里）
                decode_result = attn_forward_decode(
                    q=q_for_ffa,
                    k_q=k_q,
                    k_scale=k_scale,
                    v=v_quantized,
                    k_current=cache_layer.k_current,
                    v_current=cache_layer.v_current,
                    current_len=current_len,
                    max_current=cache_layer.max_current,
                    k_residual=k_residual,
                    **decode_kwargs,
                )
            else:
                # ========== Buffer 已初始化，使用 CUDA Graph ==========

                # 🆕 获取当前 buffer shape
                current_buffer_shape = (
                    cache_layer.k_q_buffer.shape,
                    cache_layer.k_scale_buffer.shape,
                    cache_layer.v_buffer.shape,
                )

                # 🆕 检查是否需要初始化 CUDA Graph
                need_init = (
                    self.cudagraph_runner is None or
                    self.cudagraph_buffer_shape != current_buffer_shape
                )

                if need_init:
                    # 延迟导入
                    try:
                        from attn_kernel.attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8
                    except ImportError:
                        import sys
                        from pathlib import Path
                        sys.path.insert(0, str(Path(__file__).parent.parent / "attn_kernel"))
                        from attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8

                    # 准备 runner 参数
                    runner_kwargs = {
                        'q': q_for_ffa,
                        'k_q': cache_layer.k_q_buffer,  # 🎯 使用 buffer（固定 shape）
                        'k_scale': cache_layer.k_scale_buffer,
                        'v': cache_layer.v_buffer,
                        'k_current': cache_layer.k_current,
                        'v_current': cache_layer.v_current,
                        'current_len': current_len,
                        'k_residual': cache_layer.k_residual_buffer,
                        'quantized_len': cache_layer.quantized_len,  # 🎯 初始有效长度
                        'k_bits': decode_kwargs.get('k_bits', 2),
                        'scale': self.scaling,
                        'BS': decode_kwargs.get('BS', 128),
                        'SBS': decode_kwargs.get('SBS', 128),
                        'delta': decode_kwargs.get('delta', 5.0),
                        'use_fp8_residual': decode_kwargs.get('use_fp8_residual', True),
                        'max_current': cache_layer.max_current,
                        'warmup': 2,  # Triton JIT warmup
                    }

                    # 创建 CUDA Graph runner
                    try:
                        self.cudagraph_runner = CUDAGraphDecodeRunnerQ2FP8(**runner_kwargs)
                        self.cudagraph_buffer_shape = current_buffer_shape
                        self.cudagraph_initialized = True
                        print(f"[Layer {self.layer_idx}] CUDA Graph initialized: "
                              f"buffer_shape={current_buffer_shape[0]}, "
                              f"quantized_len={cache_layer.quantized_len}")
                    except Exception as e:
                        print(f"[Layer {self.layer_idx}] CUDA Graph init failed: {e}")
                        # Fallback 到标准路径
                        self.cudagraph_runner = None
                        self.cudagraph_initialized = False

                # 🆕 使用 CUDA Graph 或 fallback
                if self.cudagraph_runner is not None:
                    # CUDA Graph 路径（快速）
                    try:
                        decode_result = self.cudagraph_runner.replay(
                            q=q_for_ffa,
                            k_q=cache_layer.k_q_buffer,  # 整个 buffer
                            k_scale=cache_layer.k_scale_buffer,
                            v=cache_layer.v_buffer,
                            k_current=cache_layer.k_current,
                            v_current=cache_layer.v_current,
                            current_len=current_len,
                            k_residual=cache_layer.k_residual_buffer,
                            quantized_len=cache_layer.quantized_len,  # 🎯 动态长度
                            return_skip_ratio=return_skip,
                        )
                    except Exception as e:
                        print(f"[Layer {self.layer_idx}] CUDA Graph replay failed: {e}, "
                              "falling back to standard path")
                        # Fallback
                        decode_result = attn_forward_decode(
                            q=q_for_ffa,
                            k_q=k_q,
                            k_scale=k_scale,
                            v=v_quantized,
                            k_current=cache_layer.k_current,
                            v_current=cache_layer.v_current,
                            current_len=current_len,
                            max_current=cache_layer.max_current,
                            k_residual=k_residual,
                            **decode_kwargs,
                        )
                else:
                    # 标准路径（fallback）
                    decode_result = attn_forward_decode(
                        q=q_for_ffa,
                        k_q=k_q,
                        k_scale=k_scale,
                        v=v_quantized,
                        k_current=cache_layer.k_current,
                        v_current=cache_layer.v_current,
                        current_len=current_len,
                        max_current=cache_layer.max_current,
                        k_residual=k_residual,
                        **decode_kwargs,
                    )

            # ... 后续处理代码保持不变 ...
            # 解析 decode_result，处理 skip_ratio 等


# ============================================================================
# 📝 使用说明
# ============================================================================
"""
完整实现步骤：

1. 复制原文件：
   cp e2e/q2fp8-unified/ffa_model/modeling_llama.py \
      e2e/q2fp8-unified-optimized/ffa_model/modeling_llama_optimized.py

2. 应用上述修改：
   - 在 LlamaAttention.__init__ 中添加 CUDA Graph 相关成员变量
   - 在 forward 方法的 decode 路径中：
     a. 检查 cache_layer.buffer_initialized
     b. 如果未初始化，使用标准路径
     c. 如果已初始化，检查是否需要创建 CUDA Graph runner
     d. 使用 runner.replay() 执行 decode

3. 导入优化的模块：
   在 E2E 测试脚本中：
   ```python
   sys.path.insert(0, "e2e/q2fp8-unified-optimized/ffa_model")
   from modeling_llama_optimized import LlamaForCausalLM as FFALlamaForCausalLM
   from q2fp8_cache_optimized import Q2FP8SymCache
   ```

4. 测试：
   python e2e/benchmark_prefill_decode.py \
       --model_path /path/to/model \
       --prompt_lengths 16384 \
       --decode_lengths 256

关键点：
- 只在 buffer_initialized=True 后才使用 CUDA Graph
- 第一次 decode 会初始化 CUDA Graph（有 ~100-200ms 延迟）
- 后续 decode 全部使用 graph replay（无延迟）
- Buffer shape 固定，quantized_len 动态变化
- 有完善的 fallback 机制，确保稳定性

预期日志输出：
[Layer 0] CUDA Graph initialized: buffer_shape=(1, 20480, 8, 32), quantized_len=16384
[Layer 1] CUDA Graph initialized: buffer_shape=(1, 20480, 8, 32), quantized_len=16384
...
（后续 decode 不再输出，直接使用 replay）

性能验证：
- 第一次 decode: ~100-200ms（包含 CUDA Graph capture）
- 后续 decode: ~10-15ms（使用 graph replay）
- 对比原版本: ~25-30ms（标准 kernel 调用）
- 预期提升: ~1.5-2x
"""

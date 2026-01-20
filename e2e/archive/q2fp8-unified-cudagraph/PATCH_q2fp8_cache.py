"""
Q2FP8 Cache 优化版本 - 预分配 Buffer 实现

关键修改点：
1. 预分配固定大小的 K/V buffer
2. 使用 O(1) copy 替代 O(n) cat
3. 支持 CUDA Graph 的固定 shape 要求

使用方法：
    from q2fp8_cache_optimized import Q2FP8SymCache

    cache = Q2FP8SymCache(
        BS=128,
        use_fp8_residual=True,
        k_bits=2,
        max_decode_tokens=4096,  # 🆕 预分配大小
    )
"""

# ============================================================================
# 🆕 关键修改 1: 在 Q2FP8SymLayer.__init__ 中添加预分配相关变量
# ============================================================================
# 位置：class Q2FP8SymLayer(CacheLayerMixin): def __init__(...)
#
# 在原有代码基础上添加：

def __init__(
    self,
    BS: int = 128,
    use_fp8_residual: bool = True,
    k_bits: int = 2,
    max_current: Optional[int] = None,
    max_decode_tokens: int = 4096,  # 🆕 新增参数
):
    # ... 原有代码 ...

    # 🆕 预分配 buffer（固定 shape）
    self.k_q_buffer: Optional[torch.Tensor] = None
    self.k_scale_buffer: Optional[torch.Tensor] = None
    self.k_residual_buffer: Optional[torch.Tensor] = None
    self.v_buffer: Optional[torch.Tensor] = None

    # 🆕 有效长度追踪
    self.quantized_len: int = 0  # 已量化的 tokens 数量
    self.value_len: int = 0      # V cache 有效长度

    # 🆕 状态标志
    self.buffer_initialized: bool = False
    self.max_decode_tokens: int = max_decode_tokens


# ============================================================================
# 🆕 关键修改 2: 添加 buffer 初始化方法
# ============================================================================
# 位置：在 Q2FP8SymLayer 类中添加新方法

def _initialize_buffers_after_prefill(
    self,
    initial_k_q: torch.Tensor,
    initial_k_scale: torch.Tensor,
    initial_k_residual: torch.Tensor,
    initial_v: torch.Tensor,
):
    """
    Prefill 后调用一次，预分配所有 decode 阶段需要的 buffer。

    Args:
        initial_k_q: [B, T_prefill, HKV, K_packed]
        initial_k_scale: [B, num_blocks_prefill, HKV, K]
        initial_k_residual: [B, T_prefill, HKV, K]
        initial_v: [B, T_prefill, HKV, V]
    """
    B, T_prefill, HKV, K_packed = initial_k_q.shape
    _, _, _, V = initial_v.shape
    K = initial_k_scale.shape[-1]

    # 计算总容量
    max_total_tokens = T_prefill + self.max_decode_tokens
    max_total_blocks = (max_total_tokens + self.BS - 1) // self.BS

    # 🆕 预分配固定大小 buffer
    self.k_q_buffer = torch.empty(
        (B, max_total_tokens, HKV, K_packed),
        device=initial_k_q.device,
        dtype=initial_k_q.dtype
    )
    self.k_scale_buffer = torch.empty(
        (B, max_total_blocks, HKV, K),
        device=initial_k_scale.device,
        dtype=initial_k_scale.dtype
    )
    self.k_residual_buffer = torch.empty(
        (B, max_total_tokens, HKV, K),
        device=initial_k_residual.device,
        dtype=initial_k_residual.dtype
    )
    self.v_buffer = torch.empty(
        (B, max_total_tokens, HKV, V),
        device=initial_v.device,
        dtype=initial_v.dtype
    )

    # 拷贝 prefill 数据到 buffer
    self.k_q_buffer[:, :T_prefill, :, :].copy_(initial_k_q)
    self.k_scale_buffer[:, :initial_k_scale.shape[1], :, :].copy_(initial_k_scale)
    self.k_residual_buffer[:, :T_prefill, :, :].copy_(initial_k_residual)
    self.v_buffer[:, :T_prefill, :, :].copy_(initial_v)

    # 设置有效长度
    self.quantized_len = T_prefill
    self.num_full_blocks = initial_k_scale.shape[1]
    self.value_len = T_prefill

    # 更新 view 指针
    self._update_views()

    print(f"[Q2FP8Cache] Buffers initialized: "
          f"prefill={T_prefill}, max_decode={self.max_decode_tokens}, "
          f"total_capacity={max_total_tokens}")


def _update_views(self):
    """更新 view 指针，指向 buffer 的有效区域（向后兼容）。"""
    if self.k_q_buffer is not None:
        self.k_q = self.k_q_buffer.narrow(self.seq_dim, 0, self.quantized_len)
        self.k_scale = self.k_scale_buffer.narrow(1, 0, self.num_full_blocks)
        self.k_residual = self.k_residual_buffer.narrow(self.seq_dim, 0, self.quantized_len)
        self.value = self.v_buffer.narrow(self.seq_dim, 0, self.value_len)


# ============================================================================
# 🆕 关键修改 3: 修改量化方法，写入 buffer 而非 cat
# ============================================================================
# 位置：修改 _quantize_and_store_blocks 方法

def _quantize_and_store_blocks(self, k_blocks, cos, sin):
    """
    量化新 blocks 并写入预分配的 buffer（不改变 shape）。
    """
    # 应用 RoPE
    if cos is not None and sin is not None:
        k_blocks = self._apply_rope(k_blocks, cos, sin)

    # 量化
    k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(
        k_blocks, block_size=self.BS, k_bits=self.k_bits
    )

    new_tokens = k_q_new.shape[self.seq_dim]
    new_blocks = k_scale_new.shape[1]

    # 🆕 检查容量
    if self.quantized_len + new_tokens > self.k_q_buffer.shape[self.seq_dim]:
        raise RuntimeError(
            f"Quantized buffer overflow: {self.quantized_len + new_tokens} > "
            f"{self.k_q_buffer.shape[self.seq_dim]}"
        )

    # 🆕 直接写入 buffer（而不是 cat）
    start = self.quantized_len
    end = start + new_tokens
    self.k_q_buffer[:, start:end, :, :].copy_(k_q_new)
    self.k_residual_buffer[:, start:end, :, :].copy_(k_residual_new)

    start_block = self.num_full_blocks
    end_block = start_block + new_blocks
    self.k_scale_buffer[:, start_block:end_block, :, :].copy_(k_scale_new)

    # 更新有效长度
    self.quantized_len += new_tokens
    self.num_full_blocks += new_blocks

    # 更新 view 指针
    self._update_views()


# ============================================================================
# 🆕 关键修改 4: 修改 update 方法，处理 V cache 和 buffer 初始化
# ============================================================================
# 位置：修改 update 方法

def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.is_initialized:
        self.lazy_initialization(key_states)

    # 提取 cos 和 sin
    cos = None
    sin = None
    if cache_kwargs is not None:
        cos = cache_kwargs.get("cos")
        sin = cache_kwargs.get("sin")

    # 🆕 检测 prefill（seq_len > 1）
    is_prefill = key_states.shape[self.seq_dim] > 1

    if is_prefill and not self.buffer_initialized:
        # ========== Prefill 阶段 ==========
        # 使用原有逻辑进行量化
        # ... 原有的 prefill 量化代码 ...
        # （这里保持原有逻辑不变）

        # 量化完成后，初始化 buffer
        self._initialize_buffers_after_prefill(
            self.k_q, self.k_scale, self.k_residual, self.value
        )
        self.buffer_initialized = True

    else:
        # ========== Decode 阶段 ==========
        # 🆕 更新 V cache 到 buffer（O(1) copy）
        new_len = value_states.shape[self.seq_dim]
        if self.value_len + new_len > self.v_buffer.shape[self.seq_dim]:
            raise RuntimeError(f"V buffer overflow")

        self.v_buffer[:, self.value_len:self.value_len + new_len, :, :].copy_(value_states)
        self.value_len += new_len

        # 更新 K cache（使用修改后的 _quantize_and_store_blocks）
        # ... 原有的 K cache 更新逻辑 ...
        # （调用 _quantize_and_store_blocks，它会写入 buffer）

    self._refresh_fp_cache()
    return key_states, value_states


# ============================================================================
# 🆕 关键修改 5: 在 Q2FP8SymCache.__init__ 中传递 max_decode_tokens
# ============================================================================
# 位置：class Q2FP8SymCache(Cache): def __init__(...)

def __init__(
    self,
    BS: int = 128,
    use_fp8_residual: bool = True,
    k_bits: int = 2,
    max_current: Optional[int] = None,
    max_decode_tokens: int = 4096,  # 🆕 新增参数
    offloading: bool = False,
    offload_only_non_sliding: bool = True,
):
    super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
    self.BS = BS
    self.use_fp8_residual = use_fp8_residual
    self.k_bits = k_bits
    self.max_current = BS if max_current is None else max_current
    self.max_decode_tokens = max_decode_tokens  # 🆕

def _ensure_layer(self, layer_idx: int) -> None:
    while len(self.layers) <= layer_idx:
        self.layers.append(
            Q2FP8SymLayer(
                BS=self.BS,
                use_fp8_residual=self.use_fp8_residual,
                k_bits=self.k_bits,
                max_current=self.max_current,
                max_decode_tokens=self.max_decode_tokens,  # 🆕 传递参数
            )
        )


# ============================================================================
# 📝 使用说明
# ============================================================================
"""
完整实现步骤：

1. 复制原文件：
   cp e2e/q2fp8-unified/ffa_model/q2fp8_cache.py \
      e2e/q2fp8-unified-optimized/ffa_model/q2fp8_cache_optimized.py

2. 应用上述修改：
   - 在 __init__ 中添加新成员变量
   - 添加 _initialize_buffers_after_prefill 方法
   - 添加 _update_views 方法
   - 修改 _quantize_and_store_blocks 方法
   - 修改 update 方法中的 V cache 更新逻辑
   - 在 Q2FP8SymCache 中传递 max_decode_tokens

3. 测试：
   python -c "from q2fp8_cache_optimized import Q2FP8SymCache; print('Import OK')"

注意事项：
- 保持原有代码的其他部分不变
- 只修改标注 🆕 的部分
- 确保 buffer 初始化只在 prefill 后执行一次
"""

"""
CUDA Graph 集成补丁 - attn_q2fp8_unified.py 修改说明

关键修改点：
1. 添加 quantized_len_tensor 参数支持动态长度
2. 修改 CUDAGraphDecodeRunnerQ2FP8 支持 buffer 和动态长度
3. 确保 kernel 使用 masking 处理有效区域

使用方法：
    from attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8

    runner = CUDAGraphDecodeRunnerQ2FP8(
        q=q,
        k_q=k_q_buffer,  # 使用 buffer（固定 shape）
        quantized_len=quantized_len,  # 初始长度
        ...
    )

    # Replay 时传递新的长度
    output = runner.replay(
        q=q,
        k_q=k_q_buffer,
        quantized_len=new_quantized_len,  # 动态长度
        ...
    )
"""

# ============================================================================
# 🆕 关键修改 1: 修改 attn_forward_decode_quantized 函数签名
# ============================================================================
# 位置：def attn_forward_decode_quantized(...) 约在 Line 608

def attn_forward_decode_quantized(
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    k_current: torch.Tensor | None = None,
    v_current: torch.Tensor | None = None,
    current_len: int = 0,
    k_residual: torch.Tensor | None = None,
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
    quantized_len_tensor: torch.Tensor | None = None,  # 🆕 新增参数
    max_current: int = 128,
    **kwargs,
):
    """
    🆕 新增参数说明：
    quantized_len_tensor: Optional[torch.Tensor]
        - Shape: [1]，标量 tensor
        - 用于在 CUDA Graph 中传递动态长度
        - 如果为 None，则从 k_q.shape[1] 推断
    """

    # ... 原有的参数检查代码 ...

    # 🆕 从 tensor 读取动态长度（如果提供）
    if quantized_len_tensor is not None:
        T = int(quantized_len_tensor.item())
        # 验证长度不超过 buffer 大小
        if T > k_q.shape[1]:
            raise ValueError(f"quantized_len ({T}) exceeds k_q buffer size ({k_q.shape[1]})")
    else:
        # 向后兼容：从 k_q shape 推断
        T = k_q.shape[1]

    # 后续代码使用 T 而不是 k_q.shape[1]
    # （原有代码中的 T = k_q.shape[1] 被替换）

    # ... 其余代码保持不变 ...
    # Kernel 会自动使用 T 进行 masking（已有逻辑）


# ============================================================================
# 🆕 关键修改 2: 修改 CUDAGraphDecodeRunnerQ2FP8.__init__
# ============================================================================
# 位置：class CUDAGraphDecodeRunnerQ2FP8: def __init__(...) 约在 Line 828

class CUDAGraphDecodeRunnerQ2FP8:
    def __init__(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,  # 🆕 现在应该是 buffer（固定 shape）
        k_scale: torch.Tensor,
        v: torch.Tensor,
        *,
        k_current: Optional[torch.Tensor] = None,
        v_current: Optional[torch.Tensor] = None,
        current_len: int = 0,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        max_kept: int | None = None,
        max_kept_ratio: float = 0.2,
        use_fp8_residual: bool = True,
        max_current: int = 128,
        quantized_len: int = None,  # 🆕 新增参数：初始有效长度
        warmup: int = 2,
        num_warps_th: Optional[int] = None,
        num_stages_th: Optional[int] = None,
        num_warps_s1: Optional[int] = None,
        num_stages_s1: Optional[int] = None,
        num_warps_s2: Optional[int] = None,
        num_stages_s2: Optional[int] = None,
    ) -> None:
        # ... 原有的初始化代码 ...

        # 🆕 推断初始有效长度
        if quantized_len is None:
            quantized_len = k_q.shape[1]

        # 🆕 创建动态长度 tensor（可在 graph 外更新）
        self._quantized_len_tensor = torch.tensor(
            [quantized_len],
            device=self._device,
            dtype=torch.int32
        )

        # 🆕 保存 buffer shape（用于验证）
        self._buffer_shape = k_q.shape

        # ... 原有的 static buffer 创建代码 ...
        # 注意：使用 k_q 的 shape（即 buffer shape）

        # Warmup（传递 quantized_len_tensor）
        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_current=self._static_k_current,
                v_current=self._static_v_current,
                current_len=self._current_len,
                quantized_len_tensor=self._quantized_len_tensor,  # 🆕 传递 tensor
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                max_current=self._max_current,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
                num_warps_s2=self._num_warps_s2,
                num_stages_s2=self._num_stages_s2,
            )
        torch.cuda.synchronize(self._device)

        # Capture graph（传递 quantized_len_tensor）
        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            self._static_out = attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_current=self._static_k_current,
                v_current=self._static_v_current,
                current_len=self._current_len,
                quantized_len_tensor=self._quantized_len_tensor,  # 🆕 传递 tensor
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                max_kept=self._max_kept,
                max_current=self._max_current,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
                num_warps_s2=self._num_warps_s2,
                num_stages_s2=self._num_stages_s2,
            )


# ============================================================================
# 🆕 关键修改 3: 修改 CUDAGraphDecodeRunnerQ2FP8.replay
# ============================================================================
# 位置：class CUDAGraphDecodeRunnerQ2FP8: def replay(...) 约在 Line 990

def replay(
    self,
    q: torch.Tensor,
    k_q: torch.Tensor,  # 🆕 应该是 buffer
    k_scale: torch.Tensor,
    v: torch.Tensor,
    *,
    k_current: Optional[torch.Tensor] = None,
    v_current: Optional[torch.Tensor] = None,
    current_len: Optional[int] = None,
    k_residual: Optional[torch.Tensor] = None,
    precomputed_threshold: Optional[torch.Tensor] = None,
    quantized_len: Optional[int] = None,  # 🆕 新增参数：当前有效长度
    return_skip_ratio: bool = False,
) -> torch.Tensor:
    # 验证设备
    if q.device != self._device:
        raise ValueError("q must be on the same device as the captured graph.")

    # 🆕 验证 buffer shape
    if k_q.shape != self._buffer_shape:
        raise ValueError(
            f"k_q shape mismatch: expected {self._buffer_shape}, got {k_q.shape}. "
            "Buffer shape must remain constant for CUDA Graph."
        )

    # 🆕 更新动态长度 tensor（在 graph 外部）
    if quantized_len is not None:
        if quantized_len > k_q.shape[1]:
            raise ValueError(f"quantized_len ({quantized_len}) exceeds buffer size ({k_q.shape[1]})")
        self._quantized_len_tensor.fill_(quantized_len)

    # 拷贝输入到 static buffers
    self._static_q.copy_(q)
    self._static_k_q.copy_(k_q)
    self._static_k_scale.copy_(k_scale)
    self._static_v.copy_(v)

    if self._use_fp8_residual:
        if k_residual is None:
            raise ValueError("k_residual is required for this captured graph.")
        self._static_k_residual.copy_(k_residual)

    if self._static_k_current is not None and k_current is not None:
        self._static_k_current.copy_(k_current)
    if self._static_v_current is not None and v_current is not None:
        self._static_v_current.copy_(v_current)

    if self._use_ext_th:
        if precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")
        self._static_threshold.copy_(precomputed_threshold)

    # 更新 current_len
    if current_len is not None:
        self._current_len = current_len

    # 🆕 Replay graph（quantized_len 已通过 tensor 更新）
    self._graph.replay()

    if not return_skip_ratio:
        return self._static_out

    # Skip ratio 需要重新运行（不在 graph 中）
    _, skip_ratio = attn_forward_decode_quantized(
        q=self._static_q,
        k_q=self._static_k_q,
        k_scale=self._static_k_scale,
        k_residual=self._static_k_residual,
        v=self._static_v,
        k_current=self._static_k_current,
        v_current=self._static_v_current,
        current_len=self._current_len,
        quantized_len_tensor=self._quantized_len_tensor,  # 🆕
        k_bits=self._k_bits,
        scale=self._scale,
        BS=self._BS,
        SBS=self._SBS,
        delta=self._delta,
        max_kept=self._max_kept,
        max_current=self._max_current,
        return_skip_ratio=True,
        precomputed_threshold=self._static_threshold,
        use_fp8_residual=self._use_fp8_residual,
        num_warps_th=self._num_warps_th,
        num_stages_th=self._num_stages_th,
        num_warps_s1=self._num_warps_s1,
        num_stages_s1=self._num_stages_s1,
        num_warps_s2=self._num_warps_s2,
        num_stages_s2=self._num_stages_s2,
    )
    return self._static_out, skip_ratio


# ============================================================================
# 📝 使用说明
# ============================================================================
"""
完整实现步骤：

1. 复制原文件：
   cp e2e/q2fp8-unified/attn_kernel/attn_q2fp8_unified.py \
      e2e/q2fp8-unified-optimized/attn_kernel/attn_q2fp8_unified_optimized.py

2. 应用上述修改：
   - 在 attn_forward_decode_quantized 中添加 quantized_len_tensor 参数
   - 修改函数内部使用 T = quantized_len_tensor.item()
   - 在 CUDAGraphDecodeRunnerQ2FP8.__init__ 中创建 _quantized_len_tensor
   - 在 warmup 和 capture 时传递 quantized_len_tensor
   - 在 replay 中更新 quantized_len_tensor 并验证 buffer shape

3. 测试：
   python -c "from attn_q2fp8_unified_optimized import CUDAGraphDecodeRunnerQ2FP8; print('Import OK')"

关键点：
- quantized_len_tensor 是一个标量 tensor，可以在 graph 外更新
- Buffer shape 必须固定，不能改变
- Kernel 内部已有 masking 逻辑（t_mask_sb = offs_t_sb < T）
- 每次 replay 只需更新 quantized_len_tensor.fill_(new_len)

验证方法：
- 检查 T 的来源：应该从 quantized_len_tensor.item() 读取
- 检查 buffer shape：应该在 __init__ 时固定
- 检查 replay：应该只更新 tensor，不重新 capture
"""

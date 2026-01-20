"""
Q2FP8 CUDA Graph Cache: 预分配固定大小 buffer + CUDA Graph 加速

核心特性:
1. 预分配固定最大长度的 buffer (避免动态内存分配)
2. 所有 tokens 统一量化存储 (无 k_current)
3. Prefill 后立即量化所有 tokens
4. Decode 阶段使用 CUDA Graph 加速

设计原则:
- 所有 tensor 形状固定,适配 CUDA Graph
- 使用 mask 标记有效长度
- Prefill 使用 flash_attn, Decode 使用 FFA + CUDA Graph
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

SUPPORTED_K_BITS = (2, 4)


def quantize_symmetric_blocks(
    k_blocks: torch.Tensor,
    block_size: int,
    k_bits: int = 2,
    eps: float = 1e-8,
):
    """
    量化多个完整 blocks 的 K。

    Args:
        k_blocks: [B, T, HKV, K]，其中 T 必须是 block_size 的整数倍
        block_size: 每个 block 的长度
        k_bits: 量化位数 (2 或 4)
        eps: 防止除零的小常数

    Returns:
        k_q: [B, T, HKV, K_packed] 量化后的 K
        k_scale: [B, num_blocks, HKV, K] 每个 block 的 scale
        k_residual: [B, T, HKV, K] FP8 残差
    """
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, T, HKV, K = k_blocks.shape
    if T % block_size != 0:
        raise ValueError(f"T={T} is not divisible by block_size={block_size}")

    dtype = k_blocks.dtype
    num_blocks = T // block_size

    # 量化参数
    QMAX = (1 << k_bits) - 1  # 2-bit: 3, 4-bit: 15
    QZERO = QMAX / 2  # 2-bit: 1.5, 4-bit: 7.5
    VALS_PER_BYTE = 8 // k_bits  # 2-bit: 4, 4-bit: 2
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    k_blocks = k_blocks.reshape(B, num_blocks, block_size, HKV, K)

    # 每个 block 的 scale (取 block 内所有 token 的最大绝对值)
    k_abs_max = k_blocks.abs().amax(dim=2)  # [B, num_blocks, HKV, K]
    k_scale = (k_abs_max / QZERO).clamp(min=eps)

    # 量化: q = round(k / scale + QZERO)
    k_norm = k_blocks / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack 到 uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_packed, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 4
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_packed)

    # 反量化用于残差
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_blocks - k_dequant
    try:
        k_residual = k_residual.to(torch.float8_e4m3fn)
    except:
        k_residual = k_residual.to(dtype)

    k_residual = k_residual.reshape(B, T, HKV, K)

    return k_q_packed, k_scale, k_residual


class Q2FP8CudaGraphLayer(CacheLayerMixin):
    """
    CUDA Graph 优化的 Q2FP8 Cache Layer。

    特点:
    - 预分配固定大小的 buffer
    - 所有 tokens 统一量化存储
    - 使用 mask 标记有效长度
    - 适配 CUDA Graph (无动态内存分配)
    """

    is_sliding = False

    def __init__(
        self,
        max_seq_len: int,
        BS: int = 128,
        use_fp8_residual: bool = True,
        k_bits: int = 2,
    ):
        super().__init__()
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len % BS != 0:
            raise ValueError(f"max_seq_len={max_seq_len} must be divisible by BS={BS}")

        self.max_seq_len = max_seq_len
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.seq_dim = 1
        self.num_blocks = max_seq_len // BS

        # 量化参数
        QMAX = (1 << k_bits) - 1
        VALS_PER_BYTE = 8 // k_bits
        self.K_packed_per_K = VALS_PER_BYTE  # 用于计算 K_packed

        # 预分配的固定大小 buffer (初始化时创建)
        self.k_q: Optional[torch.Tensor] = None           # [B, max_seq_len, HKV, K_packed]
        self.k_scale: Optional[torch.Tensor] = None       # [B, num_blocks, HKV, K]
        self.k_residual: Optional[torch.Tensor] = None    # [B, max_seq_len, HKV, K]
        self.value: Optional[torch.Tensor] = None         # [B, max_seq_len, HKV, V]

        # 当前有效长度
        self.current_len: int = 0

        # 用于兼容 transformers 接口
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """延迟初始化: 根据实际输入形状创建固定大小的 buffer"""
        self.dtype, self.device = key_states.dtype, key_states.device
        B, T, HKV, K = key_states.shape
        _, _, _, V = value_states.shape

        K_packed = (K + self.K_packed_per_K - 1) // self.K_packed_per_K

        # 预分配固定大小的 buffer
        self.k_q = torch.zeros(
            (B, self.max_seq_len, HKV, K_packed),
            dtype=torch.uint8,
            device=self.device
        )
        self.k_scale = torch.zeros(
            (B, self.num_blocks, HKV, K),
            dtype=torch.float32,
            device=self.device
        )
        if self.use_fp8_residual:
            try:
                self.k_residual = torch.zeros(
                    (B, self.max_seq_len, HKV, K),
                    dtype=torch.float8_e4m3fn,
                    device=self.device
                )
            except:
                self.k_residual = torch.zeros(
                    (B, self.max_seq_len, HKV, K),
                    dtype=self.dtype,
                    device=self.device
                )
        else:
            self.k_residual = None

        self.value = torch.zeros(
            (B, self.max_seq_len, HKV, V),
            dtype=self.dtype,
            device=self.device
        )

        self.is_initialized = True

    def _quantize_and_store(self, k_new: torch.Tensor, start_idx: int) -> None:
        """
        量化新的 tokens 并存储到预分配的 buffer 中。

        Args:
            k_new: [B, T_new, HKV, K] 新增的 K tokens
            start_idx: 存储的起始位置
        """
        B, T_new, HKV, K = k_new.shape
        end_idx = start_idx + T_new

        if end_idx > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_idx} exceeds max_seq_len {self.max_seq_len}"
            )

        # 确保 T_new 是 BS 的整数倍 (padding if needed)
        if T_new % self.BS != 0:
            pad_len = self.BS - (T_new % self.BS)
            k_new = torch.nn.functional.pad(k_new, (0, 0, 0, 0, 0, pad_len), value=0.0)
            T_new = k_new.shape[1]
            end_idx = start_idx + T_new

        # 量化
        k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(
            k_new,
            block_size=self.BS,
            k_bits=self.k_bits,
        )

        # 存储到预分配的 buffer
        self.k_q[:, start_idx:end_idx, :, :] = k_q_new

        # 计算 block 索引
        start_block = start_idx // self.BS
        num_new_blocks = k_scale_new.shape[1]
        end_block = start_block + num_new_blocks
        self.k_scale[:, start_block:end_block, :, :] = k_scale_new

        if self.k_residual is not None:
            if k_residual_new.dtype != self.k_residual.dtype:
                k_residual_new = k_residual_new.to(self.k_residual.dtype)
            self.k_residual[:, start_idx:end_idx, :, :] = k_residual_new

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        更新 cache。

        Prefill 阶段: 存储所有 tokens 并立即量化
        Decode 阶段: 逐个添加 token 并量化
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, T_new, HKV, K = key_states.shape
        start_idx = self.current_len
        end_idx = start_idx + T_new

        if end_idx > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_idx} exceeds max_seq_len {self.max_seq_len}"
            )

        # 存储 V (不量化)
        self.value[:, start_idx:end_idx, :, :] = value_states

        # 量化并存储 K
        self._quantize_and_store(key_states, start_idx)

        # 更新当前长度
        self.current_len = end_idx

        # 返回完整的 K 和 V (用于 prefill 阶段的 flash_attn)
        # 注意: decode 阶段不会使用这个返回值,而是直接使用量化的 buffer
        self.keys = self._get_dequantized_keys()
        self.values = self.value[:, :self.current_len, :, :]

        return self.keys, self.values

    def _get_dequantized_keys(self) -> torch.Tensor:
        """反量化 keys (仅用于 prefill 阶段)"""
        if self.current_len == 0:
            return None

        B, _, HKV, K_packed = self.k_q.shape
        K = self.k_scale.shape[-1]
        QMAX = (1 << self.k_bits) - 1
        QZERO = QMAX / 2

        # Unpack k_q
        k_q_slice = self.k_q[:, :self.current_len, :, :]
        if self.k_bits == 2:
            k_unpacked = torch.stack([
                (k_q_slice >> 0) & 0x3,
                (k_q_slice >> 2) & 0x3,
                (k_q_slice >> 4) & 0x3,
                (k_q_slice >> 6) & 0x3,
            ], dim=-1).view(B, self.current_len, HKV, -1)[..., :K].float()
        else:
            k_unpacked = torch.stack([
                (k_q_slice >> 0) & 0xF,
                (k_q_slice >> 4) & 0xF,
            ], dim=-1).view(B, self.current_len, HKV, -1)[..., :K].float()

        # 反量化
        # k_scale: [B, num_blocks, HKV, K]
        # 需要 expand 到 [B, current_len, HKV, K]
        num_current_blocks = (self.current_len + self.BS - 1) // self.BS
        k_scale_expanded = self.k_scale[:, :num_current_blocks, :, :].repeat_interleave(
            self.BS, dim=1
        )[:, :self.current_len, :, :]

        k_dequant = (k_unpacked - QZERO) * k_scale_expanded

        if self.k_residual is not None:
            k_residual_slice = self.k_residual[:, :self.current_len, :, :]
            k_dequant = k_dequant + k_residual_slice.to(k_dequant.dtype)

        return k_dequant.to(self.dtype)

    def get_seq_length(self) -> int:
        return self.current_len

    def get_quantized_len(self) -> int:
        return self.current_len

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """返回 mask 大小 (kv_length, kv_offset)"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.current_len + query_length
        return kv_length, kv_offset

    def reset(self) -> None:
        """重置 cache (清空但保留预分配的 buffer)"""
        if not self.is_initialized:
            return
        self.current_len = 0
        # 不需要清零 buffer,只需重置长度即可

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        if self.k_q is not None:
            self.k_q = self.k_q.repeat_interleave(repeats, dim=0)
            self.k_scale = self.k_scale.repeat_interleave(repeats, dim=0)
            if self.k_residual is not None:
                self.k_residual = self.k_residual.repeat_interleave(repeats, dim=0)
        if self.value is not None:
            self.value = self.value.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.device)
        if self.k_q is not None:
            self.k_q = self.k_q.index_select(0, indices)
            self.k_scale = self.k_scale.index_select(0, indices)
            if self.k_residual is not None:
                self.k_residual = self.k_residual.index_select(0, indices)
        if self.value is not None:
            self.value = self.value.index_select(0, indices)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class Q2FP8CudaGraphCache(Cache):
    """
    CUDA Graph 优化的 Q2FP8 Cache。

    特点:
    - 预分配固定大小的 buffer
    - 所有 tokens 统一量化存储
    - Decode 阶段使用 CUDA Graph 加速
    """

    def __init__(
        self,
        max_seq_len: int,
        BS: int = 128,
        use_fp8_residual: bool = True,
        k_bits: int = 2,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len % BS != 0:
            raise ValueError(f"max_seq_len={max_seq_len} must be divisible by BS={BS}")

        self.max_seq_len = max_seq_len
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                Q2FP8CudaGraphLayer(
                    max_seq_len=self.max_seq_len,
                    BS=self.BS,
                    use_fp8_residual=self.use_fp8_residual,
                    k_bits=self.k_bits,
                )
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)

        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def get_seq_length(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_seq_length()

    def get_quantized_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_quantized_len()

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

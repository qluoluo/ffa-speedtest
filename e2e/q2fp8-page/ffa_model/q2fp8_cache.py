"""
Q2FP8 Symmetric Cache: 对称量化 K cache + FP8 残差。

采用 Page-wise 量化策略：
- 按 block/page 独立量化，每个 block 有独立的 scale
- 新增 token 累积到当前 block，满了才量化
- 未满的 block 保持 FP16，不量化

支持的量化位数:
- 2-bit: 4 个值 packed 到 1 个 uint8, QMAX=3, QZERO=1.5
- 4-bit: 2 个值 packed 到 1 个 uint8, QMAX=15, QZERO=7.5

对称量化公式:
- scale = abs_max / QZERO
- q = round(k / scale + QZERO)
- dequant = (q - QZERO) * scale

数据布局:
- k_q: [B, num_full_blocks * BS, HKV, K_packed] 已量化的完整 blocks
- k_scale: [B, num_full_blocks, HKV, K] 每个 block 的 scale
- k_residual: [B, num_full_blocks * BS, HKV, K] FP8 残差
- k_current: [B, current_len, HKV, K] 当前未满 block，保持 FP16
- v: [B, T, HKV, V] 完整 V cache

MODIFICATION NOTICE:
此版本为极致显存优化版。
1. 移除了所有反量化逻辑。
2. update() 返回空张量 (shape=[B, 0, ...])。
   - 作用：支持 modeling_llama.py 中的 .transpose() 操作不报错。
   - 效果：若误入 Flash Attention 路径，因 seqlen=0 会触发 Runtime Error，确保只走 FFA 路径。
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

SUPPORTED_K_BITS = (2, 4)


def quantize_symmetric(k: torch.Tensor, k_bits: int = 2, eps: float = 1e-8):
    """
    对 K 进行对称量化。

    Args:
        k: [B, T, HKV, K] FP16/BF16 K tensor
        k_bits: 量化位数 (2 或 4)
        eps: 防止除零的小常数

    Returns:
        k_q: [B, T, HKV, K_packed] 量化后的 K (packed into uint8)
        k_scale: [B, HKV, K] 量化 scale
        k_residual: [B, T, HKV, K] FP8 残差
    """
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")

    B, T, HKV, K = k.shape
    dtype = k.dtype

    # 量化参数
    QMAX = (1 << k_bits) - 1  # 2-bit: 3, 4-bit: 15
    QZERO = QMAX / 2  # 2-bit: 1.5, 4-bit: 7.5
    VALS_PER_BYTE = 8 // k_bits  # 2-bit: 4, 4-bit: 2
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    # 计算每个 head 每个 dim 的 scale (取所有 token 的最大绝对值)
    k_abs_max = k.abs().amax(dim=1)  # [B, HKV, K]
    k_scale = k_abs_max / QZERO  # [B, HKV, K]
    k_scale = k_scale.clamp(min=eps)

    # 量化: q = round(k / scale + QZERO), 并 clamp 到 [0, QMAX]
    k_norm = k / k_scale.unsqueeze(1)  # [B, T, HKV, K]
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack 值到 uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, T, HKV, K_packed, VALS_PER_BYTE)

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

    # 计算反量化值用于残差
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(1)

    # 计算残差并转为 FP8
    k_residual = k - k_dequant
    try:
        k_residual = k_residual.to(torch.float8_e4m3fn)
    except:
        k_residual = k_residual.to(dtype)

    return k_q_packed, k_scale, k_residual


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
        k_q: [B, T, HKV, K_packed]
        k_scale: [B, num_blocks, HKV, K] 每个 block 的 scale
        k_residual: [B, T, HKV, K]
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


class Q2FP8SymLayer(CacheLayerMixin):
    """
    Page-wise 对称量化 Cache Layer。

    特点：
    - 按 block/page 独立量化，每个 block 有独立的 scale
    - 新增 token 累积到当前 block，满了才量化
    - 未满的 block 保持 FP16，不量化
    - 使用预分配的固定大小 buffer，避免 torch.cat() 导致地址变化
    """

    is_sliding = False

    def __init__(self, BS: int = 128, use_fp8_residual: bool = True, k_bits: int = 2, max_seq_len: int = 32768):
        super().__init__()
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.seq_dim = 1
        self.max_seq_len = max_seq_len
        self.max_blocks = (max_seq_len + BS - 1) // BS

        # 已量化的完整 blocks - 预分配固定大小
        self.k_q: Optional[torch.Tensor] = None           # [B, max_blocks * BS, HKV, K_packed]
        self.k_scale: Optional[torch.Tensor] = None       # [B, max_blocks, HKV, K]
        self.k_residual: Optional[torch.Tensor] = None    # [B, max_blocks * BS, HKV, K]
        self.num_full_blocks: int = 0                     # 实际已量化的 block 数量

        # 当前未满的 block，保持 FP16 - 使用固定大小 buffer + mask
        self.k_current: Optional[torch.Tensor] = None     # [B, BS, HKV, K] - 固定大小
        self.v_current: Optional[torch.Tensor] = None     # [B, BS, HKV, V] - 固定大小
        self.current_len: int = 0                         # 当前有效长度 (0 <= current_len <= BS)

        # 完整 V cache - 预分配固定大小
        self.value: Optional[torch.Tensor] = None         # [B, max_seq_len, HKV, V]
        self.value_len: int = 0                           # V cache 的实际有效长度

        # 用于兼容 transformers 接口
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.key_full: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """预分配所有固定大小的 buffer。"""
        self.dtype, self.device = key_states.dtype, key_states.device
        B, _, HKV, K = key_states.shape
        V = value_states.shape[-1]

        # 计算 packed K 的大小
        VALS_PER_BYTE = 8 // self.k_bits
        K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

        # 预分配 V cache - 固定大小
        self.value = torch.zeros(
            (B, self.max_seq_len, HKV, V),
            dtype=self.dtype,
            device=self.device,
        )
        self.value_len = 0

        # 预分配量化 K cache - 固定大小
        max_quantized_len = self.max_blocks * self.BS
        self.k_q = torch.zeros(
            (B, max_quantized_len, HKV, K_packed),
            dtype=torch.uint8,
            device=self.device,
        )
        self.k_scale = torch.zeros(
            (B, self.max_blocks, HKV, K),
            dtype=torch.float32,
            device=self.device,
        )
        # FP8 residual
        try:
            self.k_residual = torch.zeros(
                (B, max_quantized_len, HKV, K),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
        except:
            self.k_residual = torch.zeros(
                (B, max_quantized_len, HKV, K),
                dtype=self.dtype,
                device=self.device,
            )
        self.num_full_blocks = 0

        # 预分配 k_current 和 v_current - 固定大小 BS
        self.k_current = torch.zeros(
            (B, self.BS, HKV, K),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_current = torch.zeros(
            (B, self.BS, HKV, V),
            dtype=self.dtype,
            device=self.device,
        )
        self.current_len = 0

        self.is_initialized = True

    def _quantize_and_store_block(self, k_block: torch.Tensor) -> None:
        """量化一个完整的 block 并存储。"""
        self._quantize_and_store_blocks(k_block)

    def _quantize_and_store_blocks(self, k_blocks: torch.Tensor) -> None:
        """量化多个完整 blocks 并存储（in-place 写入预分配的 buffer）。"""
        k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(
            k_blocks,
            block_size=self.BS,
            k_bits=self.k_bits,
        )

        num_new_blocks = k_scale_new.shape[1]
        new_tokens = num_new_blocks * self.BS

        # 检查是否超出预分配的空间
        if self.num_full_blocks + num_new_blocks > self.max_blocks:
            raise RuntimeError(
                f"Cache overflow: trying to store {self.num_full_blocks + num_new_blocks} blocks, "
                f"but max_blocks={self.max_blocks}. Increase max_seq_len (current: {self.max_seq_len})"
            )

        # In-place 写入预分配的 buffer（地址不变）
        start_token = self.num_full_blocks * self.BS
        end_token = start_token + new_tokens
        start_block = self.num_full_blocks
        end_block = start_block + num_new_blocks

        self.k_q[:, start_token:end_token, :, :] = k_q_new
        self.k_scale[:, start_block:end_block, :, :] = k_scale_new

        # Handle FP8 residual
        if self.k_residual.dtype == torch.float8_e4m3fn and k_residual_new.dtype != torch.float8_e4m3fn:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new.to(torch.float8_e4m3fn)
        elif self.k_residual.dtype != torch.float8_e4m3fn and k_residual_new.dtype == torch.float8_e4m3fn:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new.to(self.k_residual.dtype)
        else:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new

        self.num_full_blocks += num_new_blocks

    def _refresh_fp_cache(self) -> None:
        """
        刷新缓存视图。
        MODIFIED: 不再反量化 Key Cache，self.keys 将为 None。
        """
        self.key_full = None
        self.keys = None
        # V Cache 仍保留用于可能的 Fallback (但如果 FlashAttn key=0，V 有值也会崩)
        self.values = self.value[:, :self.value_len, :, :] if self.value_len > 0 else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_len = key_states.shape[self.seq_dim]

        # 检查是否超出预分配的空间
        if self.value_len + new_len > self.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: trying to store {self.value_len + new_len} tokens, "
                f"but max_seq_len={self.max_seq_len}"
            )

        # 更新 V cache - in-place 写入
        self.value[:, self.value_len:self.value_len + new_len, :, :] = value_states
        self.value_len += new_len

        # 更新 K cache - in-place 写入到 k_current
        # 先把新的 key 写入 k_current
        if self.current_len + new_len <= self.BS:
            # 新 token 可以完全放入当前 block
            self.k_current[:, self.current_len:self.current_len + new_len, :, :] = key_states
            self.v_current[:, self.current_len:self.current_len + new_len, :, :] = value_states
            self.current_len += new_len
        else:
            # 需要处理跨 block 的情况
            # 先填满当前 block
            remaining_in_block = self.BS - self.current_len
            if remaining_in_block > 0:
                self.k_current[:, self.current_len:self.BS, :, :] = key_states[:, :remaining_in_block, :, :]
                self.v_current[:, self.current_len:self.BS, :, :] = value_states[:, :remaining_in_block, :, :]

            # 量化当前满的 block
            self._quantize_and_store_blocks(self.k_current.contiguous())

            # 处理剩余的 token
            remaining_keys = key_states[:, remaining_in_block:, :, :]
            remaining_values = value_states[:, remaining_in_block:, :, :]
            remaining_len = remaining_keys.shape[self.seq_dim]

            # 如果剩余的 token 超过一个 block，继续量化
            while remaining_len >= self.BS:
                block_keys = remaining_keys[:, :self.BS, :, :]
                self._quantize_and_store_blocks(block_keys.contiguous())
                remaining_keys = remaining_keys[:, self.BS:, :, :]
                remaining_values = remaining_values[:, self.BS:, :, :]
                remaining_len = remaining_keys.shape[self.seq_dim]

            # 把剩余的 token 放入新的 k_current
            self.k_current.zero_()
            self.v_current.zero_()
            if remaining_len > 0:
                self.k_current[:, :remaining_len, :, :] = remaining_keys
                self.v_current[:, :remaining_len, :, :] = remaining_values
            self.current_len = remaining_len

        # 检查当前 block 是否满了
        if self.current_len == self.BS:
            self._quantize_and_store_blocks(self.k_current.contiguous())
            self.k_current.zero_()
            self.v_current.zero_()
            self.current_len = 0

        self._refresh_fp_cache()
        
        # MODIFIED RETURN:
        # 为了不触发 OOM，我们不返回全量 Keys/Values。
        # 为了让 modeling_llama.py 中的 .transpose() 能够通过（避免 None 导致的 AttributeError），
        # 我们返回 shape=[B, 0, HKV, K] 的空 Tensor。
        # 
        # 效果：
        # 1. FFA 路径：正常运行（因为它不使用这里的 key_states，而是读内部 k_q）。
        # 2. Flash Attention 路径：因为 Key/Value 长度为 0，Flash Attention 
        #    会抛出 RuntimeError (或者 CUDA 错误)，达到“显式报错”的目的。
        B, _, HKV, K = key_states.shape
        V = value_states.shape[-1]
        
        empty_k = torch.empty((B, 0, HKV, K), dtype=self.dtype, device=self.device)
        empty_v = torch.empty((B, 0, HKV, V), dtype=self.dtype, device=self.device)
        
        return empty_k, empty_v

    def get_seq_length(self) -> int:
        quantized_len = self.num_full_blocks * self.BS
        return quantized_len + self.current_len

    def get_quantized_len(self) -> int:
        return self.num_full_blocks * self.BS

    def get_current_len(self) -> int:
        return self.current_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def reset(self) -> None:
        if not self.is_initialized:
            return
        # 重置长度，但保留预分配的 buffer（地址不变）
        self.num_full_blocks = 0
        self.current_len = 0
        self.value_len = 0
        if self.k_q is not None:
            self.k_q.zero_()
        if self.k_scale is not None:
            self.k_scale.zero_()
        if self.k_residual is not None:
            self.k_residual.zero_()
        if self.k_current is not None:
            self.k_current.zero_()
        if self.v_current is not None:
            self.v_current.zero_()
        if self.value is not None:
            self.value.zero_()
        self.key_full = None
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        # 注意：这会改变 batch size，需要重新分配 buffer
        if self.k_q is not None:
            self.k_q = self.k_q.repeat_interleave(repeats, dim=0)
            self.k_scale = self.k_scale.repeat_interleave(repeats, dim=0)
            self.k_residual = self.k_residual.repeat_interleave(repeats, dim=0)
        if self.k_current is not None:
            self.k_current = self.k_current.repeat_interleave(repeats, dim=0)
        if self.v_current is not None:
            self.v_current = self.v_current.repeat_interleave(repeats, dim=0)
        if self.value is not None:
            self.value = self.value.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.device)
        # 注意：这会改变 batch size，需要重新分配 buffer
        if self.k_q is not None:
            self.k_q = self.k_q.index_select(0, indices)
            self.k_scale = self.k_scale.index_select(0, indices)
            self.k_residual = self.k_residual.index_select(0, indices)
        if self.k_current is not None:
            self.k_current = self.k_current.index_select(0, indices)
        if self.v_current is not None:
            self.v_current = self.v_current.index_select(0, indices)
        if self.value is not None:
            self.value = self.value.index_select(0, indices)
        self._refresh_fp_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class Q2FP8SymCache(Cache):
    """
    Page-wise 对称量化 Cache。

    特点：
    - 按 block/page 独立量化，每个 block 有独立的 scale
    - 新增 token 累积到当前 block，满了才量化
    - 未满的 block 保持 FP16，不量化
    - 使用预分配的固定大小 buffer，避免地址变化（支持 CUDA Graph）
    """

    def __init__(
        self,
        BS: int = 128,
        use_fp8_residual: bool = True,
        k_bits: int = 2,
        max_seq_len: int = 32768,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.max_seq_len = max_seq_len

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                Q2FP8SymLayer(
                    BS=self.BS,
                    use_fp8_residual=self.use_fp8_residual,
                    k_bits=self.k_bits,
                    max_seq_len=self.max_seq_len,
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

    def get_current_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_current_len()
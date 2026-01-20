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


def quantize_block(k_block: torch.Tensor, k_bits: int = 2, eps: float = 1e-8):
    """
    量化单个 block 的 K。

    Returns:
        k_q: [B, block_len, HKV, K_packed]
        k_scale: [B, 1, HKV, K] 该 block 的 scale (带 block 维度方便拼接)
        k_residual: [B, block_len, HKV, K]
    """
    k_q, k_scale, k_residual = quantize_symmetric(k_block, k_bits, eps)
    k_scale = k_scale.unsqueeze(1)  # [B, HKV, K] -> [B, 1, HKV, K]
    return k_q, k_scale, k_residual


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
    """

    is_sliding = False

    def __init__(self, BS: int = 128, use_fp8_residual: bool = True, k_bits: int = 2):
        super().__init__()
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.seq_dim = 1

        # 已量化的完整 blocks
        self.k_q: Optional[torch.Tensor] = None           # [B, num_full_blocks * BS, HKV, K_packed]
        self.k_scale: Optional[torch.Tensor] = None       # [B, num_full_blocks, HKV, K]
        self.k_residual: Optional[torch.Tensor] = None    # [B, num_full_blocks * BS, HKV, K]
        self.num_full_blocks: int = 0

        # 当前未满的 block，保持 FP16 - 使用固定大小 buffer + mask
        self.k_current: Optional[torch.Tensor] = None     # [B, BS, HKV, K] - 固定大小
        self.v_current: Optional[torch.Tensor] = None     # [B, BS, HKV, V] - 固定大小
        self.current_len: int = 0                         # 当前有效长度 (0 <= current_len <= BS)

        # 完整 V cache
        self.value: Optional[torch.Tensor] = None         # [B, T, HKV, V]

        # 用于兼容 transformers 接口
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.key_full: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def _quantize_and_store_block(self, k_block: torch.Tensor) -> None:
        """量化一个完整的 block 并存储。"""
        self._quantize_and_store_blocks(k_block)

    def _quantize_and_store_blocks(self, k_blocks: torch.Tensor) -> None:
        """量化多个完整 blocks 并存储。"""
        k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(
            k_blocks,
            block_size=self.BS,
            k_bits=self.k_bits,
        )

        if self.k_q is None:
            self.k_q = k_q_new
            self.k_scale = k_scale_new
            self.k_residual = k_residual_new
        else:
            self.k_q = torch.cat([self.k_q, k_q_new], dim=self.seq_dim)
            self.k_scale = torch.cat([self.k_scale, k_scale_new], dim=1)
            # Convert float8 to float16/bfloat16 for cat, then back to float8
            if self.k_residual.dtype == torch.float8_e4m3fn:
                k_residual_fp = self.k_residual.to(torch.float16)
                k_residual_new_fp = k_residual_new.to(torch.float16)
                self.k_residual = torch.cat([k_residual_fp, k_residual_new_fp], dim=self.seq_dim).to(torch.float8_e4m3fn)
            else:
                self.k_residual = torch.cat([self.k_residual, k_residual_new], dim=self.seq_dim)

        self.num_full_blocks += k_scale_new.shape[1]

    def _dequantize_blocks(self) -> Optional[torch.Tensor]:
        """反量化所有已存储的 blocks。"""
        if self.k_q is None:
            return None

        B, T_quantized, HKV, K_packed = self.k_q.shape
        QMAX = (1 << self.k_bits) - 1
        QZERO = QMAX / 2
        K = self.k_scale.shape[-1]

        # Unpack k_q
        if self.k_bits == 2:
            k_unpacked = torch.stack([
                (self.k_q >> 0) & 0x3,
                (self.k_q >> 2) & 0x3,
                (self.k_q >> 4) & 0x3,
                (self.k_q >> 6) & 0x3,
            ], dim=-1).view(B, T_quantized, HKV, -1)[..., :K].float()
        else:
            k_unpacked = torch.stack([
                (self.k_q >> 0) & 0xF,
                (self.k_q >> 4) & 0xF,
            ], dim=-1).view(B, T_quantized, HKV, -1)[..., :K].float()

        # 反量化
        k_scale_expanded = self.k_scale.repeat_interleave(self.BS, dim=1)[:, :T_quantized, :, :]
        k_dequant = (k_unpacked - QZERO) * k_scale_expanded

        if self.k_residual is not None:
            k_dequant = k_dequant + self.k_residual.to(k_dequant.dtype)

        return k_dequant.to(self.dtype)

    def _get_full_key(self) -> Optional[torch.Tensor]:
        """获取完整的 FP16 K。"""
        parts = []
        if self.k_q is not None and self.num_full_blocks > 0:
            parts.append(self._dequantize_blocks())
        if self.k_current is not None:
            parts.append(self.k_current)
        if not parts:
            return None
        return torch.cat(parts, dim=self.seq_dim)

    def _refresh_fp_cache(self) -> None:
        self.key_full = self._get_full_key()
        self.keys = self.key_full
        self.values = self.value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # 更新 V cache
        if self.value is None:
            self.value = value_states
        else:
            self.value = torch.cat([self.value, value_states], dim=self.seq_dim)

        # 更新 K cache
        if self.k_current is None:
            self.k_current = key_states
        else:
            self.k_current = torch.cat([self.k_current, key_states], dim=self.seq_dim)

        # 当前 block 满了就量化 (可一次处理多个完整 blocks)
        if self.k_current is not None:
            full_len = self.k_current.shape[self.seq_dim]
            num_full_blocks = full_len // self.BS
            if num_full_blocks > 0:
                full_tokens = num_full_blocks * self.BS
                k_blocks = self.k_current.narrow(self.seq_dim, 0, full_tokens)
                self._quantize_and_store_blocks(k_blocks.contiguous())

                remaining_len = full_len - full_tokens
                if remaining_len > 0:
                    self.k_current = self.k_current.narrow(self.seq_dim, full_tokens, remaining_len).contiguous()
                else:
                    self.k_current = None

        self._refresh_fp_cache()
        return self.keys, self.values

    def get_seq_length(self) -> int:
        quantized_len = self.num_full_blocks * self.BS
        current_len = 0 if self.k_current is None else self.k_current.shape[self.seq_dim]
        return quantized_len + current_len

    def get_quantized_len(self) -> int:
        return self.num_full_blocks * self.BS

    def get_current_len(self) -> int:
        return 0 if self.k_current is None else self.k_current.shape[self.seq_dim]

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
        self.k_q = None
        self.k_scale = None
        self.k_residual = None
        self.num_full_blocks = 0
        self.k_current = None
        self.key_full = None
        if self.value is not None:
            self.value = self.value.narrow(self.seq_dim, 0, 0).contiguous()
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        if self.k_q is not None:
            self.k_q = self.k_q.repeat_interleave(repeats, dim=0)
            self.k_scale = self.k_scale.repeat_interleave(repeats, dim=0)
            self.k_residual = self.k_residual.repeat_interleave(repeats, dim=0)
        if self.k_current is not None:
            self.k_current = self.k_current.repeat_interleave(repeats, dim=0)
        if self.value is not None:
            self.value = self.value.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.device)
        if self.k_q is not None:
            self.k_q = self.k_q.index_select(0, indices)
            self.k_scale = self.k_scale.index_select(0, indices)
            self.k_residual = self.k_residual.index_select(0, indices)
        if self.k_current is not None:
            self.k_current = self.k_current.index_select(0, indices)
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
    """

    def __init__(
        self,
        BS: int = 128,
        use_fp8_residual: bool = True,
        k_bits: int = 2,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                Q2FP8SymLayer(
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

    def get_current_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_current_len()

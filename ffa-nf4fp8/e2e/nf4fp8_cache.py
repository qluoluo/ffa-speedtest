from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


NF4_CODEBOOK_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5229921340942383,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

_NF4_TABLE_CACHE = {}


def _get_nf4_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index)
    if key in _NF4_TABLE_CACHE:
        return _NF4_TABLE_CACHE[key]
    codebook = torch.tensor(NF4_CODEBOOK_VALUES, device=device, dtype=torch.float32)
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    _NF4_TABLE_CACHE[key] = (codebook, boundaries)
    return codebook, boundaries


def resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def encode_k_nf4_fp8_residual(
    k: torch.Tensor,
    fp8_dtype: torch.dtype,
    use_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    codebook, boundaries = _get_nf4_tables(k.device)
    k_f = k.to(torch.float32)
    scale = k_f.abs().amax(dim=1).clamp_min(1e-6)
    scale_out = scale.to(k.dtype).contiguous()
    norm = (k_f / scale[:, None, :, :]).clamp(codebook[0].item(), codebook[-1].item())
    idx = torch.bucketize(norm, boundaries).to(torch.int64)
    k_nf4 = idx.to(torch.uint8).contiguous()
    k_dequant = codebook[idx] * scale[:, None, :, :]
    k_residual = None
    if use_residual:
        k_residual = (k_f - k_dequant).to(fp8_dtype).contiguous()

    B, T, HKV, K = k_nf4.shape
    values_per_byte = 2
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_nf4.device, dtype=k_nf4.dtype)
        k_nf4 = torch.cat([k_nf4, pad_tensor], dim=-1)
    k_nf4 = k_nf4.view(B, T, HKV, k_packed_len, values_per_byte)
    k_nf4_packed = (k_nf4[..., 0] | (k_nf4[..., 1] << 4)).contiguous()
    return k_nf4_packed, scale_out, k_residual


class NF4Fp8DynamicLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, use_fp8_residual: bool = True, fp8_dtype: torch.dtype | None = None):
        super().__init__()
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype
        self.seq_dim = 1

        self.key_full: Optional[torch.Tensor] = None
        self.key_nf4: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_residual: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        if self.fp8_dtype is None:
            self.fp8_dtype = resolve_fp8_dtype(self.device)
        self.is_initialized = True

    def _append(self, stored: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if stored is None:
            return new
        return torch.cat([stored, new], dim=self.seq_dim)

    def _refresh_views(self) -> None:
        if self.key_full is None:
            self.keys = None
            self.values = None
            return
        self.keys = self.key_full
        self.values = self.value

    def _requantize_full(self) -> None:
        if self.key_full is None or self.key_full.numel() == 0:
            self.key_nf4 = None
            self.key_scale = None
            self.key_residual = None
            return
        k_nf4, k_scale, k_residual = encode_k_nf4_fp8_residual(
            self.key_full,
            self.fp8_dtype,
            use_residual=self.use_fp8_residual,
        )
        self.key_nf4 = k_nf4
        self.key_scale = k_scale
        self.key_residual = k_residual

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.key_full = self._append(self.key_full, key_states)
        self.value = self._append(self.value, value_states)
        self._requantize_full()
        self._refresh_views()

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.key_full is None or self.key_full.numel() == 0:
            return 0
        return self.key_full.shape[self.seq_dim]

    def get_max_cache_shape(self) -> int:
        return -1

    def _slice_along_seq(self, tensor: Optional[torch.Tensor], max_length: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.narrow(self.seq_dim, 0, max_length)

    def crop(self, max_length: int) -> None:
        current_len = self.get_seq_length()
        if max_length < 0:
            max_length = current_len - abs(max_length)
        if current_len == 0 or current_len <= max_length:
            return

        self.key_full = self._slice_along_seq(self.key_full, max_length)
        self.value = self._slice_along_seq(self.value, max_length)
        self._requantize_full()
        self._refresh_views()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.key_full = self.key_full.repeat_interleave(repeats, dim=0)
        self.value = self.value.repeat_interleave(repeats, dim=0)
        if self.key_nf4 is not None:
            self.key_nf4 = self.key_nf4.repeat_interleave(repeats, dim=0)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.repeat_interleave(repeats, dim=0)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        self._refresh_views()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, indices)
        self.value = self.value.index_select(0, indices)
        if self.key_nf4 is not None:
            self.key_nf4 = self.key_nf4.index_select(0, indices)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, indices)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, indices)
        self._refresh_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, beam_idx)
        self.value = self.value.index_select(0, beam_idx)
        if self.key_nf4 is not None:
            self.key_nf4 = self.key_nf4.index_select(0, beam_idx)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, beam_idx)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, beam_idx)
        self._refresh_views()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        if self.key_full is not None:
            self.key_full = self.key_full.narrow(self.seq_dim, 0, 0).contiguous()
        if self.value is not None:
            self.value = self.value.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_nf4 is not None:
            self.key_nf4 = self.key_nf4.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_residual is not None:
            self.key_residual = self.key_residual.narrow(self.seq_dim, 0, 0).contiguous()
        self.key_scale = None
        self._refresh_views()

    def offload(self):
        if self.is_initialized:
            if self.key_full is not None:
                self.key_full = self.key_full.to("cpu", non_blocking=True)
            if self.value is not None:
                self.value = self.value.to("cpu", non_blocking=True)
            if self.key_nf4 is not None:
                self.key_nf4 = self.key_nf4.to("cpu", non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to("cpu", non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.key_full is not None and self.key_full.device != self.device:
            self.key_full = self.key_full.to(self.device, non_blocking=True)
            if self.value is not None:
                self.value = self.value.to(self.device, non_blocking=True)
            if self.key_nf4 is not None:
                self.key_nf4 = self.key_nf4.to(self.device, non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to(self.device, non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
        super().prefetch()


class NF4Fp8Cache(Cache):
    def __init__(
        self,
        max_seq_len: int | None = None,
        use_fp8_residual: bool = True,
        fp8_dtype: torch.dtype | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive when provided")
        self.max_seq_len = max_seq_len
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            if self.max_seq_len is None:
                self.layers.append(
                    NF4Fp8DynamicLayer(use_fp8_residual=self.use_fp8_residual, fp8_dtype=self.fp8_dtype)
                )
            else:
                self.layers.append(
                    NF4Fp8StaticLayer(
                        max_seq_len=self.max_seq_len,
                        use_fp8_residual=self.use_fp8_residual,
                        fp8_dtype=self.fp8_dtype,
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


class NF4Fp8StaticLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, max_seq_len: int, use_fp8_residual: bool = True, fp8_dtype: torch.dtype | None = None):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype
        self.seq_dim = 1
        self.seq_len = 0

        self.key_full: Optional[torch.Tensor] = None
        self.value_full: Optional[torch.Tensor] = None
        self.key_nf4_full: Optional[torch.Tensor] = None
        self.key_residual_full: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None

        self.key_nf4: Optional[torch.Tensor] = None
        self.key_residual: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        if self.fp8_dtype is None:
            self.fp8_dtype = resolve_fp8_dtype(self.device)
        self.is_initialized = True

        batch_size, seq_len, hkv, k_dim = key_states.shape
        _, _, _, v_dim = value_states.shape
        if self.max_seq_len < seq_len:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) must be >= incoming seq_len ({seq_len})."
            )

        k_packed = (k_dim + 1) // 2
        self.key_full = torch.empty(
            (batch_size, self.max_seq_len, hkv, k_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.value_full = torch.empty(
            (batch_size, self.max_seq_len, hkv, v_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.key_nf4_full = torch.empty(
            (batch_size, self.max_seq_len, hkv, k_packed),
            device=self.device,
            dtype=torch.uint8,
        )
        if self.use_fp8_residual:
            self.key_residual_full = torch.empty(
                (batch_size, self.max_seq_len, hkv, k_dim),
                device=self.device,
                dtype=self.fp8_dtype,
            )
        self._refresh_views()

    def _refresh_views(self) -> None:
        if self.key_full is None:
            self.keys = None
            self.values = None
            self.key_nf4 = None
            self.key_residual = None
            return
        self.keys = self.key_full.narrow(self.seq_dim, 0, self.seq_len)
        self.values = self.value_full.narrow(self.seq_dim, 0, self.seq_len)
        self.key_nf4 = self.key_nf4_full.narrow(self.seq_dim, 0, self.seq_len)
        if self.use_fp8_residual and self.key_residual_full is not None:
            self.key_residual = self.key_residual_full.narrow(self.seq_dim, 0, self.seq_len)
        else:
            self.key_residual = None

    def _requantize_full(self) -> None:
        if self.seq_len <= 0:
            self.key_scale = None
            self._refresh_views()
            return
        key_slice = self.key_full.narrow(self.seq_dim, 0, self.seq_len)
        k_nf4, k_scale, k_residual = encode_k_nf4_fp8_residual(
            key_slice,
            self.fp8_dtype,
            use_residual=self.use_fp8_residual,
        )
        self.key_nf4_full[:, : self.seq_len] = k_nf4
        self.key_scale = k_scale
        if self.use_fp8_residual and self.key_residual_full is not None:
            self.key_residual_full[:, : self.seq_len] = k_residual
        self._refresh_views()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if cache_kwargs is None or cache_kwargs.get("cache_position") is None:
            raise ValueError("cache_position is required for NF4Fp8StaticLayer updates.")
        cache_position = cache_kwargs["cache_position"]
        positions = cache_position.to(device=key_states.device, dtype=torch.long)

        if positions.numel() != key_states.shape[1]:
            raise ValueError("cache_position length must match input sequence length.")

        self.key_full[:, positions] = key_states
        self.value_full[:, positions] = value_states

        if torch.cuda.is_current_stream_capturing():
            self.seq_len += key_states.shape[1]
        else:
            new_len = int(positions.max().item()) + 1
            if new_len > self.seq_len:
                self.seq_len = new_len
        self._requantize_full()
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.seq_len - abs(max_length)
        if self.seq_len <= max_length:
            return
        self.seq_len = max_length
        self._requantize_full()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.seq_len == 0:
            return
        self.key_full = self.key_full.repeat_interleave(repeats, dim=0)
        self.value_full = self.value_full.repeat_interleave(repeats, dim=0)
        self.key_nf4_full = self.key_nf4_full.repeat_interleave(repeats, dim=0)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.repeat_interleave(repeats, dim=0)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        self._refresh_views()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.seq_len == 0:
            return
        indices = indices.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, indices)
        self.value_full = self.value_full.index_select(0, indices)
        self.key_nf4_full = self.key_nf4_full.index_select(0, indices)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.index_select(0, indices)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, indices)
        self._refresh_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.seq_len == 0:
            return
        beam_idx = beam_idx.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, beam_idx)
        self.value_full = self.value_full.index_select(0, beam_idx)
        self.key_nf4_full = self.key_nf4_full.index_select(0, beam_idx)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.index_select(0, beam_idx)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, beam_idx)
        self._refresh_views()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.seq_len = 0
        self.key_scale = None
        self._refresh_views()

    def offload(self):
        if self.is_initialized:
            self.key_full = self.key_full.to("cpu", non_blocking=True)
            self.value_full = self.value_full.to("cpu", non_blocking=True)
            self.key_nf4_full = self.key_nf4_full.to("cpu", non_blocking=True)
            if self.key_residual_full is not None:
                self.key_residual_full = self.key_residual_full.to("cpu", non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.key_full.device != self.device:
            self.key_full = self.key_full.to(self.device, non_blocking=True)
            self.value_full = self.value_full.to(self.device, non_blocking=True)
            self.key_nf4_full = self.key_nf4_full.to(self.device, non_blocking=True)
            if self.key_residual_full is not None:
                self.key_residual_full = self.key_residual_full.to(self.device, non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
        super().prefetch()


class NF4Fp8StaticCache(Cache):
    def __init__(
        self,
        max_seq_len: int,
        use_fp8_residual: bool = True,
        fp8_dtype: torch.dtype | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                NF4Fp8StaticLayer(
                    max_seq_len=self.max_seq_len,
                    use_fp8_residual=self.use_fp8_residual,
                    fp8_dtype=self.fp8_dtype,
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

    def reset(self) -> None:
        if not self.layers:
            return
        for layer in self.layers:
            layer.reset()

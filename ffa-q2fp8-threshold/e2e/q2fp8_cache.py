from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def quantize_k_2bit_fp8_residual(
    k: torch.Tensor,
    fp8_dtype: torch.dtype,
    use_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
    k_residual = None
    if use_residual:
        k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    B, T, HKV, K = k_q.shape
    values_per_byte = 4
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)
        k_q = torch.cat([k_q, pad_tensor], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()
    return k_q_packed, scale, zero, k_residual


class Q2Fp8DynamicLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, use_fp8_residual: bool = True, fp8_dtype: torch.dtype | None = None):
        super().__init__()
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype
        self.seq_dim = 1

        self.key_full: Optional[torch.Tensor] = None
        self.key_q: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_zero: Optional[torch.Tensor] = None
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
            self.key_q = None
            self.key_scale = None
            self.key_zero = None
            self.key_residual = None
            return
        k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(
            self.key_full,
            self.fp8_dtype,
            use_residual=self.use_fp8_residual,
        )
        self.key_q = k_q
        self.key_scale = k_scale
        self.key_zero = k_zero
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
        if self.key_q is not None:
            self.key_q = self.key_q.repeat_interleave(repeats, dim=0)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.repeat_interleave(repeats, dim=0)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.repeat_interleave(repeats, dim=0)
        self._refresh_views()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, indices)
        self.value = self.value.index_select(0, indices)
        if self.key_q is not None:
            self.key_q = self.key_q.index_select(0, indices)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, indices)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, indices)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.index_select(0, indices)
        self._refresh_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, beam_idx)
        self.value = self.value.index_select(0, beam_idx)
        if self.key_q is not None:
            self.key_q = self.key_q.index_select(0, beam_idx)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, beam_idx)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, beam_idx)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.index_select(0, beam_idx)
        self._refresh_views()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        if self.key_full is not None:
            self.key_full = self.key_full.narrow(self.seq_dim, 0, 0).contiguous()
        if self.value is not None:
            self.value = self.value.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_q is not None:
            self.key_q = self.key_q.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_residual is not None:
            self.key_residual = self.key_residual.narrow(self.seq_dim, 0, 0).contiguous()
        self.key_scale = None
        self.key_zero = None
        self._refresh_views()

    def offload(self):
        if self.is_initialized:
            if self.key_full is not None:
                self.key_full = self.key_full.to("cpu", non_blocking=True)
            if self.value is not None:
                self.value = self.value.to("cpu", non_blocking=True)
            if self.key_q is not None:
                self.key_q = self.key_q.to("cpu", non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to("cpu", non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
            if self.key_zero is not None:
                self.key_zero = self.key_zero.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.key_full is not None and self.key_full.device != self.device:
            self.key_full = self.key_full.to(self.device, non_blocking=True)
            if self.value is not None:
                self.value = self.value.to(self.device, non_blocking=True)
            if self.key_q is not None:
                self.key_q = self.key_q.to(self.device, non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to(self.device, non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
            if self.key_zero is not None:
                self.key_zero = self.key_zero.to(self.device, non_blocking=True)
        super().prefetch()


class Q2Fp8Cache(Cache):
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
                    Q2Fp8DynamicLayer(use_fp8_residual=self.use_fp8_residual, fp8_dtype=self.fp8_dtype)
                )
            else:
                self.layers.append(
                    Q2Fp8StaticLayer(
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


class Q2Fp8StaticLayer(CacheLayerMixin):
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
        self.key_q_full: Optional[torch.Tensor] = None
        self.key_residual_full: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_zero: Optional[torch.Tensor] = None

        self.key_q: Optional[torch.Tensor] = None
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

        k_packed = (k_dim + 3) // 4
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
        self.key_q_full = torch.empty(
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
            self.key_q = None
            self.key_residual = None
            return
        self.keys = self.key_full.narrow(self.seq_dim, 0, self.seq_len)
        self.values = self.value_full.narrow(self.seq_dim, 0, self.seq_len)
        self.key_q = self.key_q_full.narrow(self.seq_dim, 0, self.seq_len)
        if self.use_fp8_residual and self.key_residual_full is not None:
            self.key_residual = self.key_residual_full.narrow(self.seq_dim, 0, self.seq_len)
        else:
            self.key_residual = None

    def _requantize_full(self) -> None:
        if self.seq_len <= 0:
            self.key_scale = None
            self.key_zero = None
            self._refresh_views()
            return
        key_slice = self.key_full.narrow(self.seq_dim, 0, self.seq_len)
        k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(
            key_slice,
            self.fp8_dtype,
            use_residual=self.use_fp8_residual,
        )
        self.key_q_full[:, : self.seq_len] = k_q
        self.key_scale = k_scale
        self.key_zero = k_zero
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
            raise ValueError("cache_position is required for Q2Fp8StaticLayer updates.")
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
        self.key_q_full = self.key_q_full.repeat_interleave(repeats, dim=0)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.repeat_interleave(repeats, dim=0)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.repeat_interleave(repeats, dim=0)
        self._refresh_views()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.seq_len == 0:
            return
        indices = indices.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, indices)
        self.value_full = self.value_full.index_select(0, indices)
        self.key_q_full = self.key_q_full.index_select(0, indices)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.index_select(0, indices)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, indices)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.index_select(0, indices)
        self._refresh_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.seq_len == 0:
            return
        beam_idx = beam_idx.to(self.key_full.device)
        self.key_full = self.key_full.index_select(0, beam_idx)
        self.value_full = self.value_full.index_select(0, beam_idx)
        self.key_q_full = self.key_q_full.index_select(0, beam_idx)
        if self.key_residual_full is not None:
            self.key_residual_full = self.key_residual_full.index_select(0, beam_idx)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, beam_idx)
        if self.key_zero is not None:
            self.key_zero = self.key_zero.index_select(0, beam_idx)
        self._refresh_views()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.seq_len = 0
        self.key_scale = None
        self.key_zero = None
        self._refresh_views()

    def offload(self):
        if self.is_initialized:
            self.key_full = self.key_full.to("cpu", non_blocking=True)
            self.value_full = self.value_full.to("cpu", non_blocking=True)
            self.key_q_full = self.key_q_full.to("cpu", non_blocking=True)
            if self.key_residual_full is not None:
                self.key_residual_full = self.key_residual_full.to("cpu", non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
            if self.key_zero is not None:
                self.key_zero = self.key_zero.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.key_full.device != self.device:
            self.key_full = self.key_full.to(self.device, non_blocking=True)
            self.value_full = self.value_full.to(self.device, non_blocking=True)
            self.key_q_full = self.key_q_full.to(self.device, non_blocking=True)
            if self.key_residual_full is not None:
                self.key_residual_full = self.key_residual_full.to(self.device, non_blocking=True)
            if self.key_scale is not None:
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
            if self.key_zero is not None:
                self.key_zero = self.key_zero.to(self.device, non_blocking=True)
        super().prefetch()


class Q2Fp8StaticCache(Cache):
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
                Q2Fp8StaticLayer(
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

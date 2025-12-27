from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _default_fp8_dtype() -> torch.dtype:
    fp8_dtype = getattr(torch, "float8_e5m2", None)
    return fp8_dtype or torch.float16


def resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def quantize_k_fp8_fp8_residual(
    k: torch.Tensor,
    fp8_dtype: torch.dtype,
    use_residual: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    k_base = k.to(fp8_dtype).contiguous()
    if not use_residual:
        return k_base, None
    k_residual = (k.to(torch.float32) - k_base.to(torch.float32)).to(fp8_dtype).contiguous()
    return k_base, k_residual


class FP8Fp8DynamicLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, use_fp8_residual: bool = True, fp8_dtype: torch.dtype | None = None):
        super().__init__()
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype or _default_fp8_dtype()
        self.seq_dim = 1

        self.key_base: Optional[torch.Tensor] = None
        self.key_residual: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def _append(self, stored: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if stored is None:
            return new
        return torch.cat([stored, new], dim=self.seq_dim)

    def _refresh_fp_cache(self) -> None:
        if self.key_base is None:
            self.keys = None
            self.values = None
            return
        keys = self.key_base.float()
        if self.use_fp8_residual and self.key_residual is not None:
            keys = keys + self.key_residual.float()
        self.keys = keys.to(dtype=self.dtype)
        self.values = self.value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        k_base, k_residual = quantize_k_fp8_fp8_residual(
            key_states, self.fp8_dtype, use_residual=self.use_fp8_residual
        )
        self.key_base = self._append(self.key_base, k_base)
        if self.use_fp8_residual:
            self.key_residual = self._append(self.key_residual, k_residual)
        self.value = self._append(self.value, value_states)
        self._refresh_fp_cache()

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.key_base is None or self.key_base.numel() == 0:
            return 0
        return self.key_base.shape[self.seq_dim]

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

        self.key_base = self._slice_along_seq(self.key_base, max_length)
        self.key_residual = self._slice_along_seq(self.key_residual, max_length)
        self.value = self._slice_along_seq(self.value, max_length)
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.key_base = self.key_base.repeat_interleave(repeats, dim=0)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.repeat_interleave(repeats, dim=0)
        self.value = self.value.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.key_base.device)
        self.key_base = self.key_base.index_select(0, indices)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, indices)
        self.value = self.value.index_select(0, indices)
        self._refresh_fp_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.key_base.device)
        self.key_base = self.key_base.index_select(0, beam_idx)
        if self.key_residual is not None:
            self.key_residual = self.key_residual.index_select(0, beam_idx)
        self.value = self.value.index_select(0, beam_idx)
        self._refresh_fp_cache()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        if self.key_base is not None:
            self.key_base = self.key_base.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_residual is not None:
            self.key_residual = self.key_residual.narrow(self.seq_dim, 0, 0).contiguous()
        if self.value is not None:
            self.value = self.value.narrow(self.seq_dim, 0, 0).contiguous()
        self._refresh_fp_cache()

    def offload(self):
        if self.is_initialized:
            if self.key_base is not None:
                self.key_base = self.key_base.to("cpu", non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to("cpu", non_blocking=True)
            if self.value is not None:
                self.value = self.value.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.keys is not None and self.keys.device != self.device:
            if self.key_base is not None:
                self.key_base = self.key_base.to(self.device, non_blocking=True)
            if self.key_residual is not None:
                self.key_residual = self.key_residual.to(self.device, non_blocking=True)
            if self.value is not None:
                self.value = self.value.to(self.device, non_blocking=True)
        super().prefetch()


class FP8Fp8Cache(Cache):
    def __init__(
        self,
        use_fp8_residual: bool = True,
        fp8_dtype: torch.dtype | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self.use_fp8_residual = use_fp8_residual
        self.fp8_dtype = fp8_dtype or _default_fp8_dtype()

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                FP8Fp8DynamicLayer(use_fp8_residual=self.use_fp8_residual, fp8_dtype=self.fp8_dtype)
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

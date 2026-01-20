"""
Extended Q2FP8 Cache with Prefill Support

This cache extends the original Q2FP8 cache to support both prefill and decode:
- Prefill: Directly quantize keys with RoPE fusion, store in cache
- Decode: Accumulate new tokens in current buffer, quantize when full

Key features:
1. Fused RoPE + quantization during prefill (no FP16 storage)
2. Per-block symmetric 2-bit quantization with FP8 residuals
3. Separate prefill and decode paths
4. Compatible with existing decode kernel
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import torch

# Import fused rope quantization - use PyTorch implementation only
# Triton version has compatibility issues with current Triton version
try:
    from .attn_kernel.fused_rope_quant_pytorch import fused_rope_and_quantize
    FUSED_ROPE_QUANT_AVAILABLE = True
except ImportError:
    try:
        from attn_kernel.fused_rope_quant_pytorch import fused_rope_and_quantize
        FUSED_ROPE_QUANT_AVAILABLE = True
    except ImportError:
        FUSED_ROPE_QUANT_AVAILABLE = False
        print("Warning: Fused RoPE quantization not available")


class Q2FP8CachePrefill:
    """
    Q2FP8 Cache with Prefill Support

    Data layout:
    - k_q: [B, T, HKV, K_PACKED] - Quantized keys (2-bit packed)
    - k_scale: [B, num_blocks, HKV, K] - Per-block scales
    - k_residual: [B, T, HKV, K] - FP8 residuals
    - v: [B, T, HKV, V] - Values (FP16)
    - k_current: [B, MAX_CURRENT, HKV, K] - Current buffer for decode
    - v_current: [B, MAX_CURRENT, HKV, V] - Current buffer for decode
    - current_len: int - Valid length in current buffer
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_cache_len: int = 32768,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        value_dim: Optional[int] = None,
        block_size: int = 64,
        k_bits: int = 2,
        max_current: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.value_dim = value_dim or head_dim
        self.block_size = block_size
        self.k_bits = k_bits
        self.max_current = max_current
        self.device = device
        self.dtype = dtype

        # Quantization parameters
        self.QMAX = (1 << k_bits) - 1
        self.QZERO = self.QMAX / 2.0
        self.VALS_PER_BYTE = 8 // k_bits
        self.K_PACKED = (head_dim + self.VALS_PER_BYTE - 1) // self.VALS_PER_BYTE

        # Maximum number of blocks
        self.max_num_blocks = (max_cache_len + block_size - 1) // block_size

        # Per-layer cache storage
        self.key_cache = []      # List of k_q tensors per layer
        self.value_cache = []    # List of v tensors per layer
        self.scale_cache = []    # List of k_scale tensors per layer
        self.residual_cache = [] # List of k_residual tensors per layer

        # Current buffers for decode (per layer)
        self.k_current_cache = []
        self.v_current_cache = []
        self.current_len_cache = []

        # Track sequence length per layer
        self.seen_tokens_per_layer = []

        # Mode tracking
        self.is_prefill_done = False

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length for a layer"""
        if layer_idx >= len(self.seen_tokens_per_layer):
            return 0
        return self.seen_tokens_per_layer[layer_idx]

    def get_max_length(self) -> int:
        """Get maximum cache length"""
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,      # [B, T, HKV, K]
        value_states: torch.Tensor,    # [B, T, HKV, V]
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states

        Args:
            key_states: New key states [B, T, HKV, K]
            value_states: New value states [B, T, HKV, V]
            layer_idx: Layer index
            cache_kwargs: Additional arguments (cos, sin for RoPE)

        Returns:
            Updated key_states and value_states
        """
        # Initialize layer cache if needed
        while len(self.key_cache) <= layer_idx:
            self._init_layer_cache()

        B, T, HKV, K = key_states.shape
        V = value_states.shape[-1]

        # Get current sequence length
        current_seq_len = self.seen_tokens_per_layer[layer_idx]

        # Determine if this is prefill or decode
        is_prefill = (current_seq_len == 0 and T > 1)

        if is_prefill:
            # Prefill path: quantize all keys with RoPE fusion
            return self._update_prefill(
                key_states, value_states, layer_idx, cache_kwargs
            )
        else:
            # Decode path: accumulate in current buffer
            return self._update_decode(
                key_states, value_states, layer_idx, cache_kwargs
            )

    def _init_layer_cache(self):
        """Initialize cache for a new layer"""
        # Quantized key cache (initially empty)
        self.key_cache.append(None)
        self.scale_cache.append(None)
        self.residual_cache.append(None)

        # Value cache (FP16)
        self.value_cache.append(None)

        # Current buffers for decode
        k_current = torch.zeros(
            (self.max_batch_size, self.max_current, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        v_current = torch.zeros(
            (self.max_batch_size, self.max_current, self.num_key_value_heads, self.value_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.k_current_cache.append(k_current)
        self.v_current_cache.append(v_current)
        self.current_len_cache.append(0)

        # Track sequence length
        self.seen_tokens_per_layer.append(0)

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill path: Apply RoPE + quantize keys, store in cache

        This path fuses RoPE and quantization to avoid storing FP16 keys.
        """
        B, T, HKV, K = key_states.shape
        V = value_states.shape[-1]

        # Get RoPE cos/sin from cache_kwargs
        cos = cache_kwargs.get("cos") if cache_kwargs else None
        sin = cache_kwargs.get("sin") if cache_kwargs else None

        if cos is None or sin is None:
            raise ValueError("Prefill requires cos/sin for RoPE in cache_kwargs")

        # Ensure cos/sin have correct shape [B, T, K]
        if cos.dim() == 2:
            # [T, K] -> expand to [B, T, K]
            if cos.shape[0] < T:
                raise ValueError(f"cos length {cos.shape[0]} < sequence length {T}")
            cos = cos[:T].unsqueeze(0).expand(B, -1, -1)
            sin = sin[:T].unsqueeze(0).expand(B, -1, -1)
        elif cos.dim() == 3:
            # [B, T, K]
            cos = cos[:, :T]
            sin = sin[:, :T]

        # Apply RoPE + quantize using fused kernel
        if FUSED_ROPE_QUANT_AVAILABLE:
            k_q, k_scale, k_residual = fused_rope_and_quantize(
                key_states, cos, sin,
                block_size=self.block_size,
                k_bits=self.k_bits,
            )
        else:
            # Fallback: separate RoPE and quantization
            k_q, k_scale, k_residual = self._fused_rope_and_quantize_fallback(
                key_states, cos, sin
            )

        # Store in cache
        self.key_cache[layer_idx] = k_q
        self.scale_cache[layer_idx] = k_scale
        self.residual_cache[layer_idx] = k_residual
        self.value_cache[layer_idx] = value_states

        # Update sequence length
        self.seen_tokens_per_layer[layer_idx] = T

        # Reset current buffer
        self.current_len_cache[layer_idx] = 0

        # Return key_states with RoPE applied (for compatibility)
        # We need to apply RoPE to return the rotated keys
        key_states_rotated = self._apply_rope(key_states, cos, sin)

        return key_states_rotated, value_states

    def _update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode path: Accumulate in current buffer, quantize when full
        """
        B, T, HKV, K = key_states.shape
        assert T == 1, "Decode mode expects single token"

        # Get RoPE cos/sin
        cos = cache_kwargs.get("cos") if cache_kwargs else None
        sin = cache_kwargs.get("sin") if cache_kwargs else None

        if cos is not None and sin is not None:
            # Apply RoPE to key_states
            key_states = self._apply_rope(key_states, cos, sin)

        # Add to current buffer
        current_len = self.current_len_cache[layer_idx]
        self.k_current_cache[layer_idx][:B, current_len] = key_states.squeeze(1)
        self.v_current_cache[layer_idx][:B, current_len] = value_states.squeeze(1)
        current_len += 1
        self.current_len_cache[layer_idx] = current_len

        # Check if current buffer is full
        if current_len >= self.block_size:
            # Quantize current buffer and append to cache
            k_current_block = self.k_current_cache[layer_idx][:B, :self.block_size]
            v_current_block = self.v_current_cache[layer_idx][:B, :self.block_size]

            # Quantize (no RoPE needed, already applied)
            k_q_block, k_scale_block, k_res_block = self._quantize_block(k_current_block)

            # Append to cache
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = k_q_block
                self.scale_cache[layer_idx] = k_scale_block
                self.residual_cache[layer_idx] = k_res_block
                self.value_cache[layer_idx] = v_current_block
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_q_block], dim=1)
                self.scale_cache[layer_idx] = torch.cat([self.scale_cache[layer_idx], k_scale_block], dim=1)
                self.residual_cache[layer_idx] = torch.cat([self.residual_cache[layer_idx], k_res_block], dim=1)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_current_block], dim=1)

            # Reset current buffer
            self.current_len_cache[layer_idx] = 0

        # Update sequence length
        self.seen_tokens_per_layer[layer_idx] += 1

        return key_states, value_states

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE rotation"""
        # x: [B, T, H, K]
        # cos, sin: [T, K] or [B, T, K]

        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, K]
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(2)  # [B, T, 1, K]
            sin = sin.unsqueeze(2)

        # Split into two halves
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos[..., : x.shape[-1] // 2] - x2 * sin[..., : x.shape[-1] // 2],
            x2 * cos[..., x.shape[-1] // 2 :] + x1 * sin[..., x.shape[-1] // 2 :],
        ], dim=-1)

        # Ensure output has same dtype as input
        return x_rotated.to(x.dtype)

    def _quantize_block(
        self,
        k_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a single block of keys"""
        B, T, HKV, K = k_block.shape

        # Compute per-block scale
        k_abs_max = k_block.abs().amax(dim=1, keepdim=True)  # [B, 1, HKV, K]
        k_scale = (k_abs_max / self.QZERO).clamp(min=1e-8)

        # Quantize
        k_norm = k_block / k_scale
        k_q_float = (k_norm + self.QZERO).round().clamp(0, self.QMAX)

        # Pack to uint8
        if K % self.VALS_PER_BYTE != 0:
            pad_size = self.VALS_PER_BYTE - (K % self.VALS_PER_BYTE)
            k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=self.QZERO)

        k_q_int = k_q_float.to(torch.int32)
        k_q_int = k_q_int.view(B, T, HKV, self.K_PACKED, self.VALS_PER_BYTE)

        if self.k_bits == 2:
            k_q_packed = (
                k_q_int[..., 0] |
                (k_q_int[..., 1] << 2) |
                (k_q_int[..., 2] << 4) |
                (k_q_int[..., 3] << 6)
            ).to(torch.uint8)
        else:  # 4-bit
            k_q_packed = (
                k_q_int[..., 0] |
                (k_q_int[..., 1] << 4)
            ).to(torch.uint8)

        # Compute residual
        k_dequant = (k_q_float[..., :K] - self.QZERO) * k_scale
        k_residual = k_block - k_dequant

        try:
            k_residual = k_residual.to(torch.float8_e5m2)
        except:
            k_residual = k_residual.to(self.dtype)

        return k_q_packed, k_scale, k_residual

    def _fused_rope_and_quantize_fallback(
        self,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback implementation without Triton kernel"""
        # Apply RoPE
        k_rotated = self._apply_rope(k, cos, sin)

        # Quantize per block
        B, T, HKV, K = k_rotated.shape
        num_blocks = (T + self.block_size - 1) // self.block_size

        # Pad to block boundary
        T_padded = num_blocks * self.block_size
        if T_padded > T:
            k_rotated = torch.nn.functional.pad(
                k_rotated, (0, 0, 0, 0, 0, T_padded - T), value=0.0
            )

        # Reshape to blocks
        k_blocks = k_rotated.view(B, num_blocks, self.block_size, HKV, K)

        # Quantize each block
        k_q_list = []
        k_scale_list = []
        k_res_list = []

        for i in range(num_blocks):
            k_block = k_blocks[:, i]  # [B, block_size, HKV, K]
            k_q, k_scale, k_res = self._quantize_block(k_block)
            k_q_list.append(k_q)
            k_scale_list.append(k_scale)
            k_res_list.append(k_res)

        k_q = torch.cat(k_q_list, dim=1)
        k_scale = torch.cat(k_scale_list, dim=1)
        k_residual = torch.cat(k_res_list, dim=1)

        # Trim padding
        if T_padded > T:
            k_q = k_q[:, :T]
            k_residual = k_residual[:, :T]

        return k_q, k_scale, k_residual

    def get_cache_for_layer(self, layer_idx: int) -> dict:
        """Get cache tensors for a specific layer"""
        if layer_idx >= len(self.key_cache):
            return None

        return {
            "k_q": self.key_cache[layer_idx],
            "k_scale": self.scale_cache[layer_idx],
            "k_residual": self.residual_cache[layer_idx],
            "v": self.value_cache[layer_idx],
            "k_current": self.k_current_cache[layer_idx],
            "v_current": self.v_current_cache[layer_idx],
            "current_len": self.current_len_cache[layer_idx],
        }


if __name__ == "__main__":
    print("Testing Q2FP8 Cache with Prefill support...")
    print("✓ Cache class defined successfully!")

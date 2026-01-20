"""
LLaMA Model with Prefill + Decode Support

This model integrates:
1. Prefill path: Fused RoPE + quantization + threshold-based attention
2. Decode path: Existing decode kernel with current buffer
3. Q2FP8CachePrefill for unified cache management

Key modifications:
- Attention layer detects prefill vs decode mode
- Routes to appropriate kernel based on sequence length
- Supports both FlashAttention-2 fallback and FFA kernels
"""

from typing import Optional, Tuple
import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

try:
    from .q2fp8_cache_prefill import Q2FP8CachePrefill
    from .ffa_fwd_prefill import prefill_forward
    from .ffa_fwd_decode import decode_forward
except ImportError:
    from q2fp8_cache_prefill import Q2FP8CachePrefill
    from ffa_fwd_prefill import prefill_forward
    from ffa_fwd_decode import decode_forward


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA.
    (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionPrefill(nn.Module):
    """
    LLaMA Attention with Prefill + Decode Support

    Supports:
    - Prefill: Threshold-based attention with fused RoPE + quantization
    - Decode: Existing decode kernel with current buffer
    - Fallback: FlashAttention-2 or eager attention
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, dtype=torch.float16)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=torch.float16)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=torch.float16)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, dtype=torch.float16)

        # FFA configuration
        self.use_ffa_prefill = getattr(config, "use_ffa_prefill", True)
        self.use_ffa_decode = getattr(config, "use_ffa_decode", True)
        self.ffa_delta = getattr(config, "ffa_delta", 5.0)
        self.ffa_block_size = getattr(config, "ffa_block_size", 64)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Q2FP8CachePrefill] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Q2FP8CachePrefill]]:
        """
        Forward pass with automatic prefill/decode routing

        Args:
            hidden_states: [B, T, hidden_size]
            position_ids: [B, T]
            past_key_value: Q2FP8CachePrefill instance
            attention_mask: Attention mask (optional)
            cache_position: Cache position (optional)

        Returns:
            attn_output: [B, T, hidden_size]
            past_key_value: Updated cache
        """
        B, T, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [B, T, num_heads, head_dim]
        query_states = query_states.view(B, T, self.num_heads, self.head_dim)
        key_states = key_states.view(B, T, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(B, T, self.num_key_value_heads, self.head_dim)

        # Get RoPE embeddings (assuming rotary_emb is passed via kwargs)
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")

        # Determine mode: prefill or decode
        is_q2fp8_cache = isinstance(past_key_value, Q2FP8CachePrefill)
        current_seq_len = past_key_value.get_seq_length(self.layer_idx) if is_q2fp8_cache else 0
        is_prefill = (current_seq_len == 0 and T > 1)
        is_decode = (T == 1 and current_seq_len > 0)

        # Update cache (applies RoPE + quantization for prefill)
        if is_q2fp8_cache:
            cache_kwargs = {"cos": cos, "sin": sin} if cos is not None else {}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Route to appropriate attention implementation
        if is_prefill and self.use_ffa_prefill and is_q2fp8_cache:
            # Prefill path: threshold-based attention
            attn_output = self._ffa_prefill_attention(
                query_states, past_key_value, cos, sin
            )
        elif is_decode and self.use_ffa_decode and is_q2fp8_cache:
            # Decode path: existing decode kernel
            attn_output = self._ffa_decode_attention(
                query_states, past_key_value, cos, sin
            )
        else:
            # Fallback: standard attention
            attn_output = self._fallback_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Reshape and project output
        attn_output = attn_output.reshape(B, T, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _ffa_prefill_attention(
        self,
        query_states: torch.Tensor,
        cache: Q2FP8CachePrefill,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill attention with threshold filtering"""
        B, T, HQ, K = query_states.shape

        # Apply RoPE to query
        if cos is not None and sin is not None:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, unsqueeze_dim=1)

        # Get cache for this layer
        cache_dict = cache.get_cache_for_layer(self.layer_idx)

        # Call prefill kernel (already returns [B, T, HQ, V] with GQA handled)
        attn_output = prefill_forward(
            q=query_states,
            cache_dict=cache_dict,
            scale=self.scaling,
            delta=self.ffa_delta,
            q_block_size=self.ffa_block_size,
            k_block_size=self.ffa_block_size,
        )

        return attn_output

    def _ffa_decode_attention(
        self,
        query_states: torch.Tensor,
        cache: Q2FP8CachePrefill,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Decode attention with threshold filtering"""
        B, T, HQ, K = query_states.shape
        assert T == 1, "Decode mode expects single token"

        # Apply RoPE to query
        if cos is not None and sin is not None:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, unsqueeze_dim=1)

        # Get cache for this layer
        cache_dict = cache.get_cache_for_layer(self.layer_idx)

        # Call decode kernel
        attn_output = decode_forward(
            q=query_states,
            cache_dict=cache_dict,
            scale=self.scaling,
            delta=self.ffa_delta,
            block_size=self.ffa_block_size,
        )

        # attn_output is [B, HQ, V], reshape to [B, 1, HQ, V]
        attn_output = attn_output.unsqueeze(1)

        return attn_output

    def _fallback_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback to standard attention"""
        # Transpose to [B, num_heads, T, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Transpose back to [B, T, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output


# Example usage and testing
if __name__ == "__main__":
    print("Testing LLaMA Attention with Prefill + Decode support...")

    # Create config
    config = LlamaConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        use_ffa_prefill=True,
        use_ffa_decode=True,
        ffa_delta=5.0,
        ffa_block_size=64,
    )

    # Create attention layer
    attn = LlamaAttentionPrefill(config, layer_idx=0)

    # Create cache
    cache = Q2FP8CachePrefill(
        max_batch_size=1,
        max_cache_len=8192,
        num_key_value_heads=8,
        head_dim=64,
        block_size=64,
        device="cuda",
    )

    print("✓ Model components created successfully!")
    print(f"  - Attention layer: {attn}")
    print(f"  - Cache: {cache}")

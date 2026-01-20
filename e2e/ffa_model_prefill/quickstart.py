"""
Quick Start Example for FFA Model Prefill

This script demonstrates basic usage of the FFA prefill + decode system.
"""

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from q2fp8_cache_prefill import Q2FP8CachePrefill
from modeling_llama_prefill import LlamaAttentionPrefill


def generate_rope_embeddings(seq_len: int, head_dim: int, device: str = "cuda"):
    """Generate simple RoPE embeddings for testing"""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    freqs = torch.outer(position_ids.squeeze(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def main():
    print("="*70)
    print("FFA Model Prefill - Quick Start Example")
    print("="*70)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cpu":
        print("Warning: Running on CPU. Performance will be slow.")

    # Model configuration
    config = LlamaConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        attention_bias=False,
        use_ffa_prefill=True,
        use_ffa_decode=True,
        ffa_delta=5.0,
        ffa_block_size=64,
    )
    config.head_dim = 64

    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  FFA delta: {config.ffa_delta}")
    print(f"  Block size: {config.ffa_block_size}")

    # Create attention layer
    print("\nCreating attention layer...")
    attn = LlamaAttentionPrefill(config, layer_idx=0).to(device)
    print("✓ Attention layer created")

    # Create cache
    print("\nCreating cache...")
    cache = Q2FP8CachePrefill(
        max_batch_size=1,
        max_cache_len=8192,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        block_size=config.ffa_block_size,
        device=device,
    )
    print("✓ Cache created")

    # Prefill phase
    print("\n" + "="*70)
    print("Phase 1: Prefill (2048 tokens)")
    print("="*70)

    prefill_len = 2048
    hidden_states = torch.randn(1, prefill_len, config.hidden_size, dtype=torch.float16, device=device)
    position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
    cos, sin = generate_rope_embeddings(prefill_len, config.head_dim, device)

    print(f"Input shape: {hidden_states.shape}")
    print("Running prefill...")

    with torch.no_grad():
        output, cache = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

    print(f"✓ Prefill complete")
    print(f"  Output shape: {output.shape}")
    print(f"  Cache length: {cache.get_seq_length(0)}")

    # Decode phase
    print("\n" + "="*70)
    print("Phase 2: Decode (10 tokens)")
    print("="*70)

    num_decode = 10
    for step in range(num_decode):
        hidden_states = torch.randn(1, 1, config.hidden_size, dtype=torch.float16, device=device)
        position_ids = torch.tensor([[prefill_len + step]], device=device)

        # Generate RoPE for current position
        cos_full, sin_full = generate_rope_embeddings(prefill_len + step + 1, config.head_dim, device)
        cos = cos_full[prefill_len + step:prefill_len + step + 1]
        sin = sin_full[prefill_len + step:prefill_len + step + 1]

        with torch.no_grad():
            output, cache = attn(hidden_states, position_ids, cache, cos=cos, sin=sin)

        if step == 0 or step == num_decode - 1:
            print(f"  Step {step}: output shape = {output.shape}, cache length = {cache.get_seq_length(0)}")

    print(f"✓ Decode complete")
    print(f"  Final cache length: {cache.get_seq_length(0)}")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Successfully processed {prefill_len + num_decode} tokens")
    print(f"✓ Prefill: {prefill_len} tokens")
    print(f"✓ Decode: {num_decode} tokens")
    print(f"✓ Final cache length: {cache.get_seq_length(0)}")

    # Cache statistics
    cache_dict = cache.get_cache_for_layer(0)
    if cache_dict and cache_dict["k_q"] is not None:
        k_q = cache_dict["k_q"]
        k_scale = cache_dict["k_scale"]
        v = cache_dict["v"]

        print(f"\nCache Statistics:")
        print(f"  Quantized keys shape: {k_q.shape}")
        print(f"  Scales shape: {k_scale.shape}")
        print(f"  Values shape: {v.shape}")
        print(f"  Current buffer length: {cache_dict['current_len']}")

        # Memory usage
        k_q_mem = k_q.numel() * k_q.element_size()
        k_scale_mem = k_scale.numel() * k_scale.element_size()
        v_mem = v.numel() * v.element_size()
        total_mem = k_q_mem + k_scale_mem + v_mem

        print(f"\nMemory Usage:")
        print(f"  Quantized keys: {k_q_mem / 1024 / 1024:.2f} MB")
        print(f"  Scales: {k_scale_mem / 1024 / 1024:.2f} MB")
        print(f"  Values: {v_mem / 1024 / 1024:.2f} MB")
        print(f"  Total: {total_mem / 1024 / 1024:.2f} MB")

        # Compare with FP16
        fp16_mem = (prefill_len + num_decode) * config.num_key_value_heads * config.head_dim * 2 * 2  # K + V
        compression_ratio = fp16_mem / total_mem
        print(f"\nCompression vs FP16:")
        print(f"  FP16 would use: {fp16_mem / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

    print("\n" + "="*70)
    print("✓ Quick start example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

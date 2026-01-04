"""
Simple example demonstrating paged Q2FP8 attention usage.
"""

import torch
from e2e.paged_q2fp8_cache import PagedQ2FP8Cache
from attn_kernel.paged_attn import paged_attn_forward_decode


def main():
    print("=" * 80)
    print("FFA Q2FP8 Paged Attention - Simple Example")
    print("=" * 80)

    # Configuration
    batch_size = 2
    num_heads_q = 32
    num_heads_kv = 8
    head_dim = 128
    page_size = 128
    num_layers = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads Q/KV: {num_heads_q}/{num_heads_kv}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Page size: {page_size}")

    # Create cache
    cache = PagedQ2FP8Cache(
        page_size=page_size,
        max_pages=1024,
        max_batch_size=batch_size,
        num_layers=num_layers,
        use_fp8_residual=True,
        device=device,
        dtype=dtype,
    )

    print(f"\n{'='*80}")
    print("Step 1: Prefill Phase")
    print(f"{'='*80}")

    # Prefill with different lengths for each batch
    prefill_lengths = [512, 768]

    for b in range(batch_size):
        prefill_len = prefill_lengths[b]
        key_states = torch.randn(
            1, prefill_len, num_heads_kv, head_dim,
            device=device, dtype=dtype
        )
        value_states = torch.randn(
            1, prefill_len, num_heads_kv, head_dim,
            device=device, dtype=dtype
        )

        keys, values = cache.update(key_states, value_states, layer_idx=0, batch_idx=b)
        print(f"  Batch {b}: Prefilled {prefill_len} tokens -> seq_len={keys.shape[0]}")

    layer0 = cache.get_layer(0)
    print(f"\n  Total physical pages allocated: {cache.next_free_page}")
    print(f"  Pages per batch: {layer0.num_pages_per_batch[:batch_size].tolist()}")

    print(f"\n{'='*80}")
    print("Step 2: Decode Phase")
    print(f"{'='*80}")

    # Decode 10 tokens
    num_decode_steps = 10

    for step in range(num_decode_steps):
        # Generate query for current step
        query = torch.randn(batch_size, 1, num_heads_q, head_dim, device=device, dtype=dtype)

        # Compute attention
        output, stats = paged_attn_forward_decode(
            q=query,
            page_table_k=layer0.page_table_k[:batch_size],
            k_pages_q=layer0.k_pages_q,
            k_pages_scale=layer0.k_pages_scale,
            k_pages_zero=layer0.k_pages_zero,
            k_pages_residual=layer0.k_pages_residual,
            v_pages=layer0.v_pages,
            seq_lens=layer0.seq_lens[:batch_size],
            page_size=page_size,
            delta=5.0,
            use_threshold_pruning=True,
            return_stats=True,
        )

        # Generate new KV for next token
        key_new = torch.randn(1, 1, num_heads_kv, head_dim, device=device, dtype=dtype)
        value_new = torch.randn(1, 1, num_heads_kv, head_dim, device=device, dtype=dtype)

        # Update cache for all batches
        for b in range(batch_size):
            cache.update(key_new, value_new, layer_idx=0, batch_idx=b)

        if step == 0:
            print(f"  Step {step}:")
            print(f"    Output shape: {output.shape}")
            print(f"    Total pages: {stats['total_pages']}")
            print(f"    Kept pages: {stats['kept_pages']}")
            print(f"    Pruned pages: {stats['pruned_pages']}")
            print(f"    Prune ratio: {stats['prune_ratio']:.2%}")

    print(f"\n  Decoded {num_decode_steps} tokens successfully!")
    print(f"  Final sequence lengths: {layer0.seq_lens[:batch_size].tolist()}")
    print(f"  Final pages per batch: {layer0.num_pages_per_batch[:batch_size].tolist()}")
    print(f"  Total physical pages: {cache.next_free_page}")

    print(f"\n{'='*80}")
    print("Step 3: Memory Statistics")
    print(f"{'='*80}")

    # Calculate memory usage
    k_q_mem = layer0.k_pages_q.element_size() * layer0.k_pages_q.numel()
    k_scale_mem = layer0.k_pages_scale.element_size() * layer0.k_pages_scale.numel()
    k_zero_mem = layer0.k_pages_zero.element_size() * layer0.k_pages_zero.numel()
    k_res_mem = (
        layer0.k_pages_residual.element_size() * layer0.k_pages_residual.numel()
        if layer0.k_pages_residual is not None
        else 0
    )
    v_mem = layer0.v_pages.element_size() * layer0.v_pages.numel()

    total_mem_mb = (k_q_mem + k_scale_mem + k_zero_mem + k_res_mem + v_mem) / (1024 ** 2)

    # Baseline memory (uncompressed FP16)
    max_seq_len = layer0.seq_lens[:batch_size].max().item()
    baseline_k_mem = batch_size * max_seq_len * num_heads_kv * head_dim * 2
    baseline_v_mem = batch_size * max_seq_len * num_heads_kv * head_dim * 2
    baseline_mem_mb = (baseline_k_mem + baseline_v_mem) / (1024 ** 2)

    print(f"  Allocated memory (all pages): {total_mem_mb:.2f} MB")
    print(f"  Baseline memory (FP16): {baseline_mem_mb:.2f} MB")
    print(f"  Effective compression: {baseline_mem_mb / total_mem_mb:.2f}x")

    # Actual used memory (only allocated pages)
    used_pages = cache.next_free_page
    k_q_mem_used = k_q_mem * used_pages / layer0.k_pages_q.shape[0]
    k_scale_mem_used = k_scale_mem * used_pages / layer0.k_pages_scale.shape[0]
    k_zero_mem_used = k_zero_mem * used_pages / layer0.k_pages_zero.shape[0]
    k_res_mem_used = k_res_mem * used_pages / layer0.k_pages_residual.shape[0] if layer0.k_pages_residual is not None else 0
    v_mem_used = v_mem * used_pages / layer0.v_pages.shape[0]

    used_mem_mb = (k_q_mem_used + k_scale_mem_used + k_zero_mem_used + k_res_mem_used + v_mem_used) / (1024 ** 2)

    print(f"\n  Used memory ({used_pages} pages): {used_mem_mb:.2f} MB")
    print(f"  Compression ratio (used): {baseline_mem_mb / used_mem_mb:.2f}x")

    print(f"\n{'='*80}")
    print("Example completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

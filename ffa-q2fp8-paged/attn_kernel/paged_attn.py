"""
Page-aware attention with Q2FP8 quantization and threshold pruning.

This is a PyTorch-based implementation for rapid prototyping.
Can be optimized to Triton kernels later for production use.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .page_quant import dequantize_k_page_q2fp8
except ImportError:
    from page_quant import dequantize_k_page_q2fp8


def paged_attn_forward_decode(
    q: torch.Tensor,
    page_table_k: torch.Tensor,
    k_pages_q: torch.Tensor,
    k_pages_scale: torch.Tensor,
    k_pages_zero: torch.Tensor,
    k_pages_residual: Optional[torch.Tensor],
    v_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    scale: Optional[float] = None,
    delta: float = 5.0,
    use_threshold_pruning: bool = True,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    Page-aware decode attention with Q2FP8 quantization and threshold pruning.

    Args:
        q: [B, 1, HQ, K]
            Query (decode, single token)
        page_table_k: [B, MAX_NUM_PAGES]
            Page table for K cache (physical page IDs)
        k_pages_q: [NUM_PHYSICAL_PAGES, HKV, PAGE_SIZE, K_packed]
            Quantized K pages (2-bit packed)
        k_pages_scale: [NUM_PHYSICAL_PAGES, HKV, K]
            Per-page per-channel scale
        k_pages_zero: [NUM_PHYSICAL_PAGES, HKV, K]
            Per-page per-channel zero point
        k_pages_residual: [NUM_PHYSICAL_PAGES, HKV, PAGE_SIZE, K], optional
            FP8 residual
        v_pages: [NUM_PHYSICAL_PAGES, HKV, PAGE_SIZE, V]
            V cache (FP16/BF16)
        seq_lens: [B]
            Actual sequence length for each batch
        page_size: int
            Tokens per page
        scale: float, optional
            Attention scale (default: 1/sqrt(K))
        delta: float
            Threshold margin for pruning (threshold = max_score - delta)
        use_threshold_pruning: bool
            Whether to use threshold-based pruning
        return_stats: bool
            Whether to return pruning statistics

    Returns:
        output: [B, HQ, V]
            Attention output
        stats (optional): dict with pruning statistics

    Notes:
        - Threshold pruning operates at page granularity
        - Threshold is computed from first and last pages
        - Pages below threshold are skipped entirely
    """
    B, _, HQ, K = q.shape
    device = q.device
    dtype = q.dtype

    # Squeeze query: [B, HQ, K]
    q = q.squeeze(1)

    # Determine HKV and V from v_pages
    NUM_PHYSICAL_PAGES, HKV, PAGE_SIZE, V = v_pages.shape
    assert PAGE_SIZE == page_size, f"Page size mismatch: {PAGE_SIZE} vs {page_size}"

    # GQA group size
    G = HQ // HKV
    assert HQ % HKV == 0, f"HQ ({HQ}) must be divisible by HKV ({HKV})"

    # Attention scale
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # Compute number of pages for each batch
    num_pages_per_batch = (seq_lens + page_size - 1) // page_size  # [B]
    max_num_pages = num_pages_per_batch.max().item()

    # Output buffer
    output = torch.zeros(B, HQ, V, device=device, dtype=torch.float32)

    # Statistics
    total_pages = 0
    pruned_pages = 0

    # Process each batch
    for b in range(B):
        num_pages = num_pages_per_batch[b].item()
        if num_pages == 0:
            continue

        total_pages += num_pages

        # Get page IDs for this batch
        page_ids = page_table_k[b, :num_pages]  # [num_pages]

        # Reshape q for this batch: [HKV, G, K]
        q_b = q[b].view(HKV, G, K)  # [HQ, K] -> [HKV, G, K]

        # --- Threshold computation (from first and last pages) ---
        threshold = None
        if use_threshold_pruning and num_pages >= 2:
            # First page
            page_id_first = page_ids[0].item()
            k_first = dequantize_k_page_q2fp8(
                k_pages_q[page_id_first],
                k_pages_scale[page_id_first],
                k_pages_zero[page_id_first],
                k_pages_residual[page_id_first] if k_pages_residual is not None else None,
                head_dim=K,
            )  # [HKV, PAGE_SIZE, K]

            # Last page
            page_id_last = page_ids[num_pages - 1].item()
            k_last = dequantize_k_page_q2fp8(
                k_pages_q[page_id_last],
                k_pages_scale[page_id_last],
                k_pages_zero[page_id_last],
                k_pages_residual[page_id_last] if k_pages_residual is not None else None,
                head_dim=K,
            )  # [HKV, PAGE_SIZE, K]

            # Compute scores for first and last pages
            # q_b: [HKV, G, K], k: [HKV, PAGE_SIZE, K]
            # scores: [HKV, G, PAGE_SIZE]
            scores_first = torch.einsum('hgk,hpk->hgp', q_b, k_first) * scale
            scores_last = torch.einsum('hgk,hpk->hgp', q_b, k_last) * scale

            # Max scores: [HKV, G]
            max_first = scores_first.amax(dim=-1)  # [HKV, G]
            max_last = scores_last.amax(dim=-1)  # [HKV, G]

            # Threshold: [HKV, G]
            threshold = torch.maximum(max_first, max_last) - delta

        # --- Process each page ---
        # Accumulate in log-space (more numerically stable)
        m_acc = torch.full((HKV, G), float('-inf'), device=device, dtype=torch.float32)  # max
        l_acc = torch.zeros(HKV, G, device=device, dtype=torch.float32)  # sum(exp)
        o_acc = torch.zeros(HKV, G, V, device=device, dtype=torch.float32)  # weighted sum

        for p_idx in range(num_pages):
            page_id = page_ids[p_idx].item()

            # Dequantize K page (without residual first, for pruning check)
            k_page_q = dequantize_k_page_q2fp8(
                k_pages_q[page_id],
                k_pages_scale[page_id],
                k_pages_zero[page_id],
                k_residual=None,  # No residual for initial check
                head_dim=K,
            )  # [HKV, PAGE_SIZE, K]

            # Compute scores: [HKV, G, PAGE_SIZE]
            scores_page = torch.einsum('hgk,hpk->hgp', q_b, k_page_q) * scale

            # Check pruning threshold
            if threshold is not None:
                max_page = scores_page.amax(dim=-1)  # [HKV, G]
                # Prune if ALL heads are below threshold
                if (max_page < threshold).all():
                    pruned_pages += 1
                    continue  # Skip this page

            # Refine with residual if available
            if k_pages_residual is not None:
                k_page_refined = dequantize_k_page_q2fp8(
                    k_pages_q[page_id],
                    k_pages_scale[page_id],
                    k_pages_zero[page_id],
                    k_residual=k_pages_residual[page_id],
                    head_dim=K,
                )
                scores_page = torch.einsum('hgk,hpk->hgp', q_b, k_page_refined) * scale

            # Handle last page (may not be full)
            if p_idx == num_pages - 1:
                actual_len = seq_lens[b].item() - (num_pages - 1) * page_size
                if actual_len < page_size:
                    # Mask out invalid tokens
                    mask = torch.arange(page_size, device=device) < actual_len
                    scores_page = scores_page.masked_fill(~mask, float('-inf'))

            # Compute page-level max
            m_page = scores_page.amax(dim=-1)  # [HKV, G]

            # Update global max
            m_new = torch.maximum(m_acc, m_page)

            # Compute softmax with stable numerics
            # exp(scores - m_new)
            p_page = torch.exp(scores_page - m_new[..., None])  # [HKV, G, PAGE_SIZE]

            # Rescale previous accumulator
            alpha = torch.exp(m_acc - m_new)  # [HKV, G]
            l_acc = l_acc * alpha  # Rescale sum(exp)
            o_acc = o_acc * alpha[..., None]  # Rescale weighted sum

            # Add current page contribution
            l_acc = l_acc + p_page.sum(dim=-1)  # [HKV, G]

            # Load V page and accumulate
            v_page = v_pages[page_id]  # [HKV, PAGE_SIZE, V]
            o_page = torch.einsum('hgp,hpv->hgv', p_page.to(dtype), v_page)
            o_acc = o_acc + o_page.to(torch.float32)

            # Update max
            m_acc = m_new

        # Normalize: o_acc / l_acc
        o_final = o_acc / l_acc[..., None].clamp_min(1e-6)  # [HKV, G, V]

        # Reshape to [HQ, V]
        o_final = o_final.reshape(HQ, V)

        # Store to output
        output[b] = o_final

    # Convert to original dtype
    output = output.to(dtype)

    if return_stats:
        stats = {
            'total_pages': total_pages,
            'pruned_pages': pruned_pages,
            'kept_pages': total_pages - pruned_pages,
            'prune_ratio': pruned_pages / total_pages if total_pages > 0 else 0.0,
        }
        return output, stats

    return output


if __name__ == "__main__":
    # Test paged attention
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    HQ = 32
    HKV = 8
    K = 128
    V = 128
    PAGE_SIZE = 128
    NUM_PAGES_TOTAL = 10  # Total physical pages

    # Create dummy data
    seq_lens = torch.tensor([256, 384], device=device)  # 2 and 3 pages
    max_num_pages = (seq_lens.max() + PAGE_SIZE - 1) // PAGE_SIZE

    # Page table
    page_table_k = torch.zeros(B, max_num_pages, device=device, dtype=torch.long)
    page_table_k[0, :2] = torch.tensor([0, 1])  # Batch 0: pages 0, 1
    page_table_k[1, :3] = torch.tensor([2, 3, 4])  # Batch 1: pages 2, 3, 4

    # Create quantized K pages
    try:
        from .page_quant import quantize_k_page_q2fp8
    except ImportError:
        from page_quant import quantize_k_page_q2fp8

    k_pages_q_list = []
    k_pages_scale_list = []
    k_pages_zero_list = []
    k_pages_residual_list = []
    v_pages_list = []

    for _ in range(NUM_PAGES_TOTAL):
        # Random K page
        k_page = torch.randn(HKV, PAGE_SIZE, K, device=device, dtype=torch.float16)
        q_packed, scale, zero, residual = quantize_k_page_q2fp8(k_page)
        k_pages_q_list.append(q_packed)
        k_pages_scale_list.append(scale)
        k_pages_zero_list.append(zero)
        k_pages_residual_list.append(residual)

        # Random V page
        v_page = torch.randn(HKV, PAGE_SIZE, V, device=device, dtype=torch.float16)
        v_pages_list.append(v_page)

    k_pages_q = torch.stack(k_pages_q_list, dim=0)  # [NUM_PAGES, HKV, PAGE_SIZE, K_packed]
    k_pages_scale = torch.stack(k_pages_scale_list, dim=0)
    k_pages_zero = torch.stack(k_pages_zero_list, dim=0)
    k_pages_residual = torch.stack(k_pages_residual_list, dim=0)
    v_pages = torch.stack(v_pages_list, dim=0)

    # Query
    q = torch.randn(B, 1, HQ, K, device=device, dtype=torch.float16)

    print(f"Query shape: {q.shape}")
    print(f"Page table shape: {page_table_k.shape}")
    print(f"K pages shape: {k_pages_q.shape}")
    print(f"V pages shape: {v_pages.shape}")
    print(f"Seq lens: {seq_lens}")

    # Run attention
    output, stats = paged_attn_forward_decode(
        q=q,
        page_table_k=page_table_k,
        k_pages_q=k_pages_q,
        k_pages_scale=k_pages_scale,
        k_pages_zero=k_pages_zero,
        k_pages_residual=k_pages_residual,
        v_pages=v_pages,
        seq_lens=seq_lens,
        page_size=PAGE_SIZE,
        delta=5.0,
        use_threshold_pruning=True,
        return_stats=True,
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Stats: {stats}")

    # Test without pruning
    output_no_prune = paged_attn_forward_decode(
        q=q,
        page_table_k=page_table_k,
        k_pages_q=k_pages_q,
        k_pages_scale=k_pages_scale,
        k_pages_zero=k_pages_zero,
        k_pages_residual=k_pages_residual,
        v_pages=v_pages,
        seq_lens=seq_lens,
        page_size=PAGE_SIZE,
        use_threshold_pruning=False,
    )

    print(f"Output (no pruning) shape: {output_no_prune.shape}")
    print(f"Max diff: {(output - output_no_prune).abs().max().item():.6f}")

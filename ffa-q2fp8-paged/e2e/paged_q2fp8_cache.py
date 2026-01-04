"""
PagedQ2FP8Cache: Page-based KV cache with Q2FP8 quantization.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from ..attn_kernel.page_quant import quantize_k_page_q2fp8, _resolve_fp8_dtype
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from attn_kernel.page_quant import quantize_k_page_q2fp8, _resolve_fp8_dtype


class PagedQ2FP8Cache:
    """
    Page-based KV cache with Q2FP8 quantization for K cache.

    Features:
        - Page-based organization (default: 128 tokens/page)
        - K cache: 2-bit + FP8 residual quantization per page
        - V cache: original precision (FP16/BF16)
        - Dynamic page allocation
        - Supports batch inference with independent page tables

    Example:
        >>> cache = PagedQ2FP8Cache(page_size=128, max_pages=2048, num_layers=32)
        >>> # During prefill/decode
        >>> cache.update(key_states, value_states, layer_idx=0, batch_idx=0)
        >>> # Access quantized data for attention
        >>> layer_cache = cache.get_layer(0)
        >>> output = paged_attn_forward_decode(
        ...     q, layer_cache.page_table_k[batch_idx],
        ...     layer_cache.k_pages_q, layer_cache.k_pages_scale, ...
        ... )
    """

    def __init__(
        self,
        page_size: int = 128,
        max_pages: int = 2048,
        max_batch_size: int = 64,
        num_layers: int = 32,
        use_fp8_residual: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize paged Q2FP8 cache.

        Args:
            page_size: Tokens per page (must match attention kernel)
            max_pages: Maximum number of physical pages to allocate
            max_batch_size: Maximum batch size
            num_layers: Number of transformer layers
            use_fp8_residual: Whether to use FP8 residual
            device: Target device (default: cuda if available)
            dtype: Data type for V cache and original K (fp16/bf16)
        """
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.use_fp8_residual = use_fp8_residual
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype

        # Resolve FP8 dtype
        self.fp8_dtype = _resolve_fp8_dtype(self.device) if use_fp8_residual else dtype

        # Per-layer cache
        self.layers: List[Optional['PagedLayerCache']] = [None] * num_layers

        # Global page allocator (simple sequential allocation)
        self.next_free_page = 0

    def allocate_page(self) -> int:
        """Allocate a new physical page."""
        if self.next_free_page >= self.max_pages:
            raise RuntimeError(f"Out of pages: max_pages={self.max_pages}")
        page_id = self.next_free_page
        self.next_free_page += 1
        return page_id

    def reset(self):
        """Reset all caches and free all pages."""
        for layer_idx in range(self.num_layers):
            if self.layers[layer_idx] is not None:
                self.layers[layer_idx].reset()
        self.next_free_page = 0

    def get_layer(self, layer_idx: int) -> 'PagedLayerCache':
        """Get or create layer cache."""
        if self.layers[layer_idx] is None:
            self.layers[layer_idx] = PagedLayerCache(
                page_size=self.page_size,
                max_pages=self.max_pages,
                max_batch_size=self.max_batch_size,
                device=self.device,
                dtype=self.dtype,
                fp8_dtype=self.fp8_dtype,
                use_fp8_residual=self.use_fp8_residual,
                parent_cache=self,
            )
        return self.layers[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            key_states: [1, seq_len, HKV, K] or [seq_len, HKV, K]
                New key states (prefill or decode)
            value_states: [1, seq_len, HKV, V] or [seq_len, HKV, V]
                New value states
            layer_idx: Layer index
            batch_idx: Batch index (for batch inference)

        Returns:
            keys: Full key cache (for compatibility)
            values: Full value cache
        """
        layer_cache = self.get_layer(layer_idx)
        return layer_cache.update(key_states, value_states, batch_idx)


class PagedLayerCache:
    """Single-layer paged cache."""

    def __init__(
        self,
        page_size: int,
        max_pages: int,
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fp8_dtype: torch.dtype,
        use_fp8_residual: bool,
        parent_cache: 'PagedQ2FP8Cache',
    ):
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_batch_size = max_batch_size
        self.device = device
        self.dtype = dtype
        self.fp8_dtype = fp8_dtype
        self.use_fp8_residual = use_fp8_residual
        self.parent_cache = parent_cache

        # Physical page storage (allocated on-demand)
        self.k_pages_q: Optional[torch.Tensor] = None  # [max_pages, HKV, PAGE_SIZE, K_packed]
        self.k_pages_scale: Optional[torch.Tensor] = None  # [max_pages, HKV, K]
        self.k_pages_zero: Optional[torch.Tensor] = None  # [max_pages, HKV, K]
        self.k_pages_residual: Optional[torch.Tensor] = None  # [max_pages, HKV, PAGE_SIZE, K]
        self.v_pages: Optional[torch.Tensor] = None  # [max_pages, HKV, PAGE_SIZE, V]

        # Page tables: [max_batch_size, MAX_PAGES_PER_BATCH]
        # MAX_PAGES_PER_BATCH will be set on first update
        self.page_table_k: Optional[torch.Tensor] = None
        self.page_table_v: Optional[torch.Tensor] = None

        # Sequence lengths: [max_batch_size]
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        # Number of pages per batch: [max_batch_size]
        self.num_pages_per_batch = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        # Temporary buffer for full keys/values (for compatibility)
        self.key_full: Dict[int, torch.Tensor] = {}
        self.value_full: Dict[int, torch.Tensor] = {}

    def _allocate_physical_pages(self, HKV: int, K: int, V: int):
        """Allocate physical page storage."""
        K_packed = (K + 3) // 4  # 4 values per byte

        self.k_pages_q = torch.zeros(
            self.max_pages, HKV, self.page_size, K_packed,
            device=self.device, dtype=torch.uint8
        )
        self.k_pages_scale = torch.zeros(
            self.max_pages, HKV, K,
            device=self.device, dtype=self.dtype
        )
        self.k_pages_zero = torch.zeros(
            self.max_pages, HKV, K,
            device=self.device, dtype=self.dtype
        )
        if self.use_fp8_residual:
            self.k_pages_residual = torch.zeros(
                self.max_pages, HKV, self.page_size, K,
                device=self.device, dtype=self.fp8_dtype
            )
        self.v_pages = torch.zeros(
            self.max_pages, HKV, self.page_size, V,
            device=self.device, dtype=self.dtype
        )

    def _allocate_page_tables(self, max_pages_per_batch: int):
        """Allocate page tables."""
        self.page_table_k = torch.zeros(
            self.max_batch_size, max_pages_per_batch,
            device=self.device, dtype=torch.long
        )
        self.page_table_v = torch.zeros(
            self.max_batch_size, max_pages_per_batch,
            device=self.device, dtype=torch.long
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a single batch.

        Args:
            key_states: [1, seq_len, HKV, K] or [seq_len, HKV, K]
            value_states: [1, seq_len, HKV, V] or [seq_len, HKV, V]
            batch_idx: Batch index

        Returns:
            keys: Full key cache
            values: Full value cache
        """
        # Remove batch dim if present
        if key_states.ndim == 4:
            key_states = key_states.squeeze(0)
        if value_states.ndim == 4:
            value_states = value_states.squeeze(0)

        seq_len, HKV, K = key_states.shape
        _, _, V = value_states.shape

        # First-time allocation
        if self.k_pages_q is None:
            self._allocate_physical_pages(HKV, K, V)

        # Append to full buffer (for compatibility)
        if batch_idx not in self.key_full:
            self.key_full[batch_idx] = key_states
            self.value_full[batch_idx] = value_states
        else:
            self.key_full[batch_idx] = torch.cat([self.key_full[batch_idx], key_states], dim=0)
            self.value_full[batch_idx] = torch.cat([self.value_full[batch_idx], value_states], dim=0)

        # Update sequence length
        total_len = self.key_full[batch_idx].shape[0]
        self.seq_lens[batch_idx] = total_len

        # Compute required pages
        num_pages_needed = (total_len + self.page_size - 1) // self.page_size

        # Allocate page table if needed
        if self.page_table_k is None:
            max_pages_per_batch = (self.max_pages + self.max_batch_size - 1) // self.max_batch_size
            self._allocate_page_tables(max_pages_per_batch)

        # Allocate new pages if needed
        current_num_pages = self.num_pages_per_batch[batch_idx].item()
        if num_pages_needed > current_num_pages:
            for _ in range(num_pages_needed - current_num_pages):
                page_id = self.parent_cache.allocate_page()
                self.page_table_k[batch_idx, current_num_pages] = page_id
                self.page_table_v[batch_idx, current_num_pages] = page_id
                current_num_pages += 1
            self.num_pages_per_batch[batch_idx] = current_num_pages

        # Quantize and store pages
        k_full = self.key_full[batch_idx]  # [total_len, HKV, K]
        v_full = self.value_full[batch_idx]  # [total_len, HKV, V]

        # Pad to multiple of page_size
        padded_len = num_pages_needed * self.page_size
        if total_len < padded_len:
            pad_len = padded_len - total_len
            k_full = torch.cat([
                k_full,
                torch.zeros(pad_len, HKV, K, device=self.device, dtype=self.dtype)
            ], dim=0)
            v_full = torch.cat([
                v_full,
                torch.zeros(pad_len, HKV, V, device=self.device, dtype=self.dtype)
            ], dim=0)

        # Reshape to pages: [num_pages, page_size, HKV, K/V]
        k_pages = k_full.view(num_pages_needed, self.page_size, HKV, K)
        v_pages = v_full.view(num_pages_needed, self.page_size, HKV, V)

        # Transpose: [num_pages, HKV, page_size, K/V]
        k_pages = k_pages.transpose(1, 2)
        v_pages = v_pages.transpose(1, 2)

        # Quantize and store each page
        for p_idx in range(num_pages_needed):
            page_id = self.page_table_k[batch_idx, p_idx].item()

            # Quantize K page
            q_packed, scale, zero, residual = quantize_k_page_q2fp8(
                k_pages[p_idx],
                fp8_dtype=self.fp8_dtype if self.use_fp8_residual else None,
            )

            # Store in physical pages
            self.k_pages_q[page_id] = q_packed
            self.k_pages_scale[page_id] = scale
            self.k_pages_zero[page_id] = zero
            if self.use_fp8_residual:
                self.k_pages_residual[page_id] = residual

            # Store V page (original precision)
            self.v_pages[page_id] = v_pages[p_idx]

        # Return full buffers for compatibility
        return self.key_full[batch_idx], self.value_full[batch_idx]

    def reset(self):
        """Reset this layer's cache."""
        self.key_full.clear()
        self.value_full.clear()
        self.seq_lens.zero_()
        self.num_pages_per_batch.zero_()
        if self.page_table_k is not None:
            self.page_table_k.zero_()
        if self.page_table_v is not None:
            self.page_table_v.zero_()


if __name__ == "__main__":
    # Test PagedQ2FP8Cache
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = PagedQ2FP8Cache(
        page_size=128,
        max_pages=100,
        max_batch_size=4,
        num_layers=2,
        use_fp8_residual=True,
        device=device,
    )

    B = 2
    HKV = 8
    K = 128
    V = 128

    print("=== Prefill Phase ===")
    # Simulate prefill (256 tokens)
    prefill_len = 256
    key_states = torch.randn(1, prefill_len, HKV, K, device=device, dtype=torch.float16)
    value_states = torch.randn(1, prefill_len, HKV, V, device=device, dtype=torch.float16)

    for b in range(B):
        keys, values = cache.update(key_states, value_states, layer_idx=0, batch_idx=b)
        print(f"Batch {b}: seq_len={keys.shape[0]}, num_pages={cache.layers[0].num_pages_per_batch[b].item()}")

    layer0 = cache.get_layer(0)
    print(f"Physical pages allocated: {cache.next_free_page}")
    print(f"K pages shape: {layer0.k_pages_q.shape}")
    print(f"V pages shape: {layer0.v_pages.shape}")

    print("\n=== Decode Phase ===")
    # Simulate decode (1 token at a time)
    for step in range(5):
        key_states = torch.randn(1, 1, HKV, K, device=device, dtype=torch.float16)
        value_states = torch.randn(1, 1, HKV, V, device=device, dtype=torch.float16)

        for b in range(B):
            keys, values = cache.update(key_states, value_states, layer_idx=0, batch_idx=b)

        if step == 0:
            print(f"Step {step}: seq_len={keys.shape[0]}, num_pages={layer0.num_pages_per_batch[0].item()}")

    print(f"Final seq_len: {layer0.seq_lens[0].item()}")
    print(f"Final num_pages per batch: {layer0.num_pages_per_batch[:B]}")
    print(f"Total physical pages allocated: {cache.next_free_page}")

    print("\n=== Test with paged attention ===")
    try:
        from attn_kernel.paged_attn import paged_attn_forward_decode
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from attn_kernel.paged_attn import paged_attn_forward_decode

    HQ = HKV * 4  # GQA with G=4
    q = torch.randn(B, 1, HQ, K, device=device, dtype=torch.float16)

    output, stats = paged_attn_forward_decode(
        q=q,
        page_table_k=layer0.page_table_k[:B],
        k_pages_q=layer0.k_pages_q,
        k_pages_scale=layer0.k_pages_scale,
        k_pages_zero=layer0.k_pages_zero,
        k_pages_residual=layer0.k_pages_residual,
        v_pages=layer0.v_pages,
        seq_lens=layer0.seq_lens[:B],
        page_size=128,
        delta=5.0,
        use_threshold_pruning=True,
        return_stats=True,
    )

    print(f"Attention output shape: {output.shape}")
    print(f"Pruning stats: {stats}")

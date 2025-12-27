# utils/flash.py
def flash_attn_compute(q_rope_1, k_rope, v, iters=50, warmup=10):
    from flash_attn import flash_attn_func
    from utils.bench import benchmark

    # flash_attn_func expects [B, T, H, D]
    if q_rope_1.ndim == 3:
        q = q_rope_1.unsqueeze(1)
    elif q_rope_1.ndim == 4:
        q = q_rope_1
    else:
        raise ValueError(f"q_rope_1 must have 3 or 4 dims, got {q_rope_1.ndim}")

    def run_flash():
        return flash_attn_func(q, k_rope, v, causal=False)

    return benchmark(run_flash, iters=iters, warmup=warmup)

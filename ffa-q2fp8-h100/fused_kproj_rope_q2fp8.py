import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    return (x * cos) + (rotate_half(x) * sin)


def quantize_k_2bit_fp8_residual(
    k: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
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


def _normalize_k_weight(w_k: torch.Tensor, in_dim: int) -> torch.Tensor:
    if w_k.ndim != 2:
        raise ValueError(f"w_k must be 2D, got {w_k.shape}")
    if w_k.shape[1] == in_dim:
        return w_k
    if w_k.shape[0] == in_dim:
        return w_k.t().contiguous()
    raise ValueError(f"w_k shape {w_k.shape} is incompatible with in_dim={in_dim}")


def _resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def _torch_to_tl_dtype(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"unsupported dtype: {dtype}")




@triton.jit
def _kproj_rope_minmax_kernel(
    h_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    bias_ptr,
    k_min_ptr,
    k_max_ptr,
    T,
    H: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PAIR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IN_DTYPE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    pair_count = K // 2
    n_t_blocks = tl.cdiv(T, BLOCK_T)
    pair_block = pid0 // n_t_blocks
    t_block = pid0 - pair_block * n_t_blocks
    if pair_block >= tl.cdiv(pair_count, BLOCK_PAIR):
        return

    base_pair = pair_block * BLOCK_PAIR
    offs_pair = base_pair + tl.arange(0, BLOCK_PAIR)
    k_even_idx = offs_pair * 2
    k_odd_idx = k_even_idx + 1
    k_mask_even = k_even_idx < K
    k_mask_odd = k_odd_idx < K

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < T

    acc_even = tl.zeros((BLOCK_T, BLOCK_PAIR), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_T, BLOCK_PAIR), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        h_ptrs = h_ptr + (pid_b * T + offs_t[:, None]) * H + offs_h[None, :]
        h_tile = tl.load(h_ptrs, mask=t_mask[:, None] & h_mask[None, :], other=0.0)

        out_even = pid_hkv * K + k_even_idx
        out_odd = pid_hkv * K + k_odd_idx
        w_even_ptrs = w_ptr + out_even[None, :] * H + offs_h[:, None]
        w_odd_ptrs = w_ptr + out_odd[None, :] * H + offs_h[:, None]
        w_even = tl.load(w_even_ptrs, mask=h_mask[:, None] & k_mask_even[None, :], other=0.0)
        w_odd = tl.load(w_odd_ptrs, mask=h_mask[:, None] & k_mask_odd[None, :], other=0.0)

        acc_even += tl.dot(h_tile, w_even, out_dtype=tl.float32)
        acc_odd += tl.dot(h_tile, w_odd, out_dtype=tl.float32)

    if HAS_BIAS:
        bias_even = tl.cast(
            tl.load(bias_ptr + k_even_idx + pid_hkv * K, mask=k_mask_even, other=0.0),
            tl.float32,
        )
        bias_odd = tl.cast(
            tl.load(bias_ptr + k_odd_idx + pid_hkv * K, mask=k_mask_odd, other=0.0),
            tl.float32,
        )
        acc_even += bias_even[None, :]
        acc_odd += bias_odd[None, :]

    k_even = tl.cast(acc_even, IN_DTYPE)
    k_odd = tl.cast(acc_odd, IN_DTYPE)
    cos_even_ptrs = cos_ptr + offs_t[:, None] * K + k_even_idx[None, :]
    sin_even_ptrs = sin_ptr + offs_t[:, None] * K + k_even_idx[None, :]
    cos_odd_ptrs = cos_ptr + offs_t[:, None] * K + k_odd_idx[None, :]
    sin_odd_ptrs = sin_ptr + offs_t[:, None] * K + k_odd_idx[None, :]
    cos_even = tl.cast(
        tl.load(cos_even_ptrs, mask=t_mask[:, None] & k_mask_even[None, :], other=0.0),
        IN_DTYPE,
    )
    sin_even = tl.cast(
        tl.load(sin_even_ptrs, mask=t_mask[:, None] & k_mask_even[None, :], other=0.0),
        IN_DTYPE,
    )
    cos_odd = tl.cast(
        tl.load(cos_odd_ptrs, mask=t_mask[:, None] & k_mask_odd[None, :], other=0.0),
        IN_DTYPE,
    )
    sin_odd = tl.cast(
        tl.load(sin_odd_ptrs, mask=t_mask[:, None] & k_mask_odd[None, :], other=0.0),
        IN_DTYPE,
    )
    k_even_rot = k_even * cos_even - k_odd * sin_even
    k_odd_rot = k_odd * cos_odd + k_even * sin_odd

    neg_inf = float("-inf")
    pos_inf = float("inf")
    even_for_max = tl.cast(
        tl.where(t_mask[:, None] & k_mask_even[None, :], k_even_rot, neg_inf),
        tl.float32,
    )
    even_for_min = tl.cast(
        tl.where(t_mask[:, None] & k_mask_even[None, :], k_even_rot, pos_inf),
        tl.float32,
    )
    odd_for_max = tl.cast(
        tl.where(t_mask[:, None] & k_mask_odd[None, :], k_odd_rot, neg_inf),
        tl.float32,
    )
    odd_for_min = tl.cast(
        tl.where(t_mask[:, None] & k_mask_odd[None, :], k_odd_rot, pos_inf),
        tl.float32,
    )

    block_max_even = tl.max(even_for_max, axis=0)
    block_min_even = -tl.max(-even_for_min, axis=0)
    block_max_odd = tl.max(odd_for_max, axis=0)
    block_min_odd = -tl.max(-odd_for_min, axis=0)

    base_k_ptr = (pid_b * HKV + pid_hkv) * K
    tl.atomic_max(k_max_ptr + base_k_ptr + k_even_idx, block_max_even, mask=k_mask_even)
    tl.atomic_min(k_min_ptr + base_k_ptr + k_even_idx, block_min_even, mask=k_mask_even)
    tl.atomic_max(k_max_ptr + base_k_ptr + k_odd_idx, block_max_odd, mask=k_mask_odd)
    tl.atomic_min(k_min_ptr + base_k_ptr + k_odd_idx, block_min_odd, mask=k_mask_odd)


@triton.jit
def _kproj_rope_quant_kernel(
    h_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    bias_ptr,
    k_min_ptr,
    k_max_ptr,
    k_q_ptr,
    k_scale_ptr,
    k_zero_ptr,
    k_res_ptr,
    T,
    H: tl.constexpr,
    K_PACKED: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PAIR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    RES_DTYPE: tl.constexpr,
    SCALE_DTYPE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    pair_count = K // 2
    n_t_blocks = tl.cdiv(T, BLOCK_T)
    pair_block = pid0 // n_t_blocks
    t_block = pid0 - pair_block * n_t_blocks
    if pair_block >= tl.cdiv(pair_count, BLOCK_PAIR):
        return

    base_pair = pair_block * BLOCK_PAIR
    offs_pair = base_pair + tl.arange(0, BLOCK_PAIR)
    k_even_idx = offs_pair * 2
    k_odd_idx = k_even_idx + 1
    k_mask_even = k_even_idx < K
    k_mask_odd = k_odd_idx < K

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < T

    base_k_ptr = (pid_b * HKV + pid_hkv) * K
    min_even_f = tl.load(k_min_ptr + base_k_ptr + k_even_idx, mask=k_mask_even, other=0.0)
    max_even_f = tl.load(k_max_ptr + base_k_ptr + k_even_idx, mask=k_mask_even, other=0.0)
    min_odd_f = tl.load(k_min_ptr + base_k_ptr + k_odd_idx, mask=k_mask_odd, other=0.0)
    max_odd_f = tl.load(k_max_ptr + base_k_ptr + k_odd_idx, mask=k_mask_odd, other=0.0)

    min_even = tl.cast(min_even_f, IN_DTYPE)
    max_even = tl.cast(max_even_f, IN_DTYPE)
    min_odd = tl.cast(min_odd_f, IN_DTYPE)
    max_odd = tl.cast(max_odd_f, IN_DTYPE)

    min_scale = tl.full((BLOCK_PAIR,), 1e-6, IN_DTYPE)
    scale_even = tl.maximum(tl.cast((max_even - min_even) / 3.0, IN_DTYPE), min_scale)
    scale_odd = tl.maximum(tl.cast((max_odd - min_odd) / 3.0, IN_DTYPE), min_scale)
    if t_block == 0:
        tl.store(k_scale_ptr + base_k_ptr + k_even_idx, tl.cast(scale_even, SCALE_DTYPE), mask=k_mask_even)
        tl.store(k_zero_ptr + base_k_ptr + k_even_idx, tl.cast(min_even, SCALE_DTYPE), mask=k_mask_even)
        tl.store(k_scale_ptr + base_k_ptr + k_odd_idx, tl.cast(scale_odd, SCALE_DTYPE), mask=k_mask_odd)
        tl.store(k_zero_ptr + base_k_ptr + k_odd_idx, tl.cast(min_odd, SCALE_DTYPE), mask=k_mask_odd)

    acc_even = tl.zeros((BLOCK_T, BLOCK_PAIR), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_T, BLOCK_PAIR), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        h_ptrs = h_ptr + (pid_b * T + offs_t[:, None]) * H + offs_h[None, :]
        h_tile = tl.load(h_ptrs, mask=t_mask[:, None] & h_mask[None, :], other=0.0)

        out_even = pid_hkv * K + k_even_idx
        out_odd = pid_hkv * K + k_odd_idx
        w_even_ptrs = w_ptr + out_even[None, :] * H + offs_h[:, None]
        w_odd_ptrs = w_ptr + out_odd[None, :] * H + offs_h[:, None]
        w_even = tl.load(w_even_ptrs, mask=h_mask[:, None] & k_mask_even[None, :], other=0.0)
        w_odd = tl.load(w_odd_ptrs, mask=h_mask[:, None] & k_mask_odd[None, :], other=0.0)

        acc_even += tl.dot(h_tile, w_even, out_dtype=tl.float32)
        acc_odd += tl.dot(h_tile, w_odd, out_dtype=tl.float32)

    if HAS_BIAS:
        bias_even = tl.cast(
            tl.load(bias_ptr + k_even_idx + pid_hkv * K, mask=k_mask_even, other=0.0),
            tl.float32,
        )
        bias_odd = tl.cast(
            tl.load(bias_ptr + k_odd_idx + pid_hkv * K, mask=k_mask_odd, other=0.0),
            tl.float32,
        )
        acc_even += bias_even[None, :]
        acc_odd += bias_odd[None, :]

    k_even = tl.cast(acc_even, IN_DTYPE)
    k_odd = tl.cast(acc_odd, IN_DTYPE)
    cos_even_ptrs = cos_ptr + offs_t[:, None] * K + k_even_idx[None, :]
    sin_even_ptrs = sin_ptr + offs_t[:, None] * K + k_even_idx[None, :]
    cos_odd_ptrs = cos_ptr + offs_t[:, None] * K + k_odd_idx[None, :]
    sin_odd_ptrs = sin_ptr + offs_t[:, None] * K + k_odd_idx[None, :]
    cos_even = tl.cast(
        tl.load(cos_even_ptrs, mask=t_mask[:, None] & k_mask_even[None, :], other=0.0),
        IN_DTYPE,
    )
    sin_even = tl.cast(
        tl.load(sin_even_ptrs, mask=t_mask[:, None] & k_mask_even[None, :], other=0.0),
        IN_DTYPE,
    )
    cos_odd = tl.cast(
        tl.load(cos_odd_ptrs, mask=t_mask[:, None] & k_mask_odd[None, :], other=0.0),
        IN_DTYPE,
    )
    sin_odd = tl.cast(
        tl.load(sin_odd_ptrs, mask=t_mask[:, None] & k_mask_odd[None, :], other=0.0),
        IN_DTYPE,
    )
    k_even_rot = k_even * cos_even - k_odd * sin_even
    k_odd_rot = k_odd * cos_odd + k_even * sin_odd

    k_even_f = tl.cast(k_even_rot, tl.float32)
    k_odd_f = tl.cast(k_odd_rot, tl.float32)
    min_even_f = tl.cast(min_even, tl.float32)
    min_odd_f = tl.cast(min_odd, tl.float32)
    scale_even_f = tl.cast(scale_even, tl.float32)
    scale_odd_f = tl.cast(scale_odd, tl.float32)
    x_even = (k_even_f - min_even_f[None, :]) / scale_even_f[None, :]
    x_odd = (k_odd_f - min_odd_f[None, :]) / scale_odd_f[None, :]
    floor_even = tl.floor(x_even)
    floor_odd = tl.floor(x_odd)
    frac_even = x_even - floor_even
    frac_odd = x_odd - floor_odd
    is_half_even = frac_even == 0.5
    is_half_odd = frac_odd == 0.5
    floor_even_i = tl.cast(floor_even, tl.int32)
    floor_odd_i = tl.cast(floor_odd, tl.int32)
    is_odd_even = (floor_even_i & 1) == 1
    is_odd_odd = (floor_odd_i & 1) == 1
    round_up_even = frac_even > 0.5
    round_up_odd = frac_odd > 0.5
    q_even = tl.where(round_up_even | (is_half_even & is_odd_even), floor_even + 1.0, floor_even)
    q_odd = tl.where(round_up_odd | (is_half_odd & is_odd_odd), floor_odd + 1.0, floor_odd)
    q_even = tl.cast(tl.minimum(tl.maximum(q_even, 0.0), 3.0), tl.int32)
    q_odd = tl.cast(tl.minimum(tl.maximum(q_odd, 0.0), 3.0), tl.int32)

    deq_even = tl.cast(q_even, tl.float32) * scale_even_f[None, :] + min_even_f[None, :]
    deq_odd = tl.cast(q_odd, tl.float32) * scale_odd_f[None, :] + min_odd_f[None, :]
    res_even = tl.cast(k_even_f - deq_even, RES_DTYPE)
    res_odd = tl.cast(k_odd_f - deq_odd, RES_DTYPE)

    base_res_ptr = (pid_b * T + offs_t[:, None]) * (HKV * K) + pid_hkv * K
    tl.store(k_res_ptr + base_res_ptr + k_even_idx[None, :], res_even, mask=t_mask[:, None] & k_mask_even[None, :])
    tl.store(k_res_ptr + base_res_ptr + k_odd_idx[None, :], res_odd, mask=t_mask[:, None] & k_mask_odd[None, :])

    q_even = tl.where(k_mask_even[None, :], q_even, 0)
    q_odd = tl.where(k_mask_odd[None, :], q_odd, 0)
    q_even_group = tl.reshape(q_even, (BLOCK_T, BLOCK_PAIR // 2, 2))
    q_odd_group = tl.reshape(q_odd, (BLOCK_T, BLOCK_PAIR // 2, 2))
    pair_pos = tl.arange(0, 2)
    even_weights = tl.cast(1 << (pair_pos * 4), tl.int32)
    odd_weights = tl.cast(1 << (pair_pos * 4 + 2), tl.int32)
    packed = tl.sum(q_even_group * even_weights + q_odd_group * odd_weights, axis=2)
    packed = tl.cast(packed, tl.uint8)

    base_pack = base_pair // 2
    pack_offs = base_pack + tl.arange(0, BLOCK_PAIR // 2)
    pack_mask = pack_offs < K_PACKED
    base_q_ptr = (pid_b * T + offs_t[:, None]) * (HKV * K_PACKED) + pid_hkv * K_PACKED
    tl.store(k_q_ptr + base_q_ptr + pack_offs[None, :], packed, mask=t_mask[:, None] & pack_mask[None, :])


def kproj_rope_quantize_reference(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(h.device)
    w_k = _normalize_k_weight(w_k, h.shape[-1])
    k_linear = F.linear(h, w_k, bias)
    B, T, _ = k_linear.shape
    if k_linear.shape[-1] != num_kv_heads * head_dim:
        raise ValueError(
            f"projection out_dim must be num_kv_heads*head_dim, got {k_linear.shape[-1]} "
            f"vs {num_kv_heads}*{head_dim}"
        )
    k = k_linear.view(B, T, num_kv_heads, head_dim)
    k = apply_rope(k, cos, sin)
    return quantize_k_2bit_fp8_residual(k, fp8_dtype)


def kproj_rope_quantize_triton(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
    block_t: int = 16,
    block_h: int = 32,
    block_pair: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not h.is_cuda:
        raise ValueError("Triton kernel requires CUDA tensors.")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    if block_pair % 2 != 0:
        raise ValueError("block_pair must be even to pack int2 values.")
    w_k = _normalize_k_weight(w_k, h.shape[-1])

    B, T, H = h.shape
    out_dim = num_kv_heads * head_dim
    if w_k.shape[0] != out_dim:
        raise ValueError(f"w_k out_dim mismatch: got {w_k.shape[0]}, expected {out_dim}")
    if bias is not None and bias.numel() != out_dim:
        raise ValueError(f"bias length mismatch: got {bias.numel()}, expected {out_dim}")

    h = h.contiguous()
    w_k = w_k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    bias_ptr = torch.empty(1, device=h.device, dtype=h.dtype)
    has_bias = False
    if bias is not None:
        bias_ptr = bias.contiguous()
        has_bias = True

    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(h.device)
    use_fp8 = hasattr(torch, "float8_e5m2") and fp8_dtype == torch.float8_e5m2
    residual_store_dtype = torch.float16 if use_fp8 else h.dtype

    K = head_dim
    HKV = num_kv_heads
    K_PACKED = (K + 3) // 4
    pair_count = K // 2

    k_min = torch.full((B, HKV, K), float("inf"), device=h.device, dtype=torch.float32)
    k_max = torch.full((B, HKV, K), float("-inf"), device=h.device, dtype=torch.float32)
    k_scale = torch.empty((B, HKV, K), device=h.device, dtype=h.dtype)
    k_zero = torch.empty((B, HKV, K), device=h.device, dtype=h.dtype)
    k_q_packed = torch.empty((B, T, HKV, K_PACKED), device=h.device, dtype=torch.uint8)
    k_residual = torch.empty((B, T, HKV, K), device=h.device, dtype=residual_store_dtype)

    n_t_blocks = triton.cdiv(T, block_t)
    n_pair_blocks = triton.cdiv(pair_count, block_pair)
    grid = (n_pair_blocks * n_t_blocks, B, HKV)

    in_tl = _torch_to_tl_dtype(h.dtype)
    res_tl = _torch_to_tl_dtype(residual_store_dtype)
    scale_tl = _torch_to_tl_dtype(h.dtype)

    _kproj_rope_minmax_kernel[grid](
        h,
        w_k,
        cos,
        sin,
        bias_ptr,
        k_min,
        k_max,
        T=T,
        H=H,
        HKV=HKV,
        K=K,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        BLOCK_PAIR=block_pair,
        HAS_BIAS=has_bias,
        IN_DTYPE=in_tl,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _kproj_rope_quant_kernel[grid](
        h,
        w_k,
        cos,
        sin,
        bias_ptr,
        k_min,
        k_max,
        k_q_packed,
        k_scale,
        k_zero,
        k_residual,
        T=T,
        H=H,
        K_PACKED=K_PACKED,
        HKV=HKV,
        K=K,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        BLOCK_PAIR=block_pair,
        HAS_BIAS=has_bias,
        IN_DTYPE=in_tl,
        RES_DTYPE=res_tl,
        SCALE_DTYPE=scale_tl,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if use_fp8:
        k_residual = k_residual.to(fp8_dtype)

    return k_q_packed, k_scale, k_zero, k_residual


def kproj_rope_quantize_fused(
    h: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
    fp8_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return kproj_rope_quantize_triton(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bias=bias,
        fp8_dtype=fp8_dtype,
    )


def run_smoke_test(
    device: torch.device,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    HKV: int,
    K: int,
) -> None:
    torch.manual_seed(0)
    h = torch.randn(B, T, H, device=device, dtype=dtype)
    w_k = torch.randn(HKV * K, H, device=device, dtype=dtype)
    cos, sin = build_rope_cache(T, K, device=device, dtype=dtype)

    ref = kproj_rope_quantize_reference(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=HKV,
        head_dim=K,
        fp8_dtype=fp8_dtype,
    )
    fused = kproj_rope_quantize_triton(
        h=h,
        w_k=w_k,
        cos=cos,
        sin=sin,
        num_kv_heads=HKV,
        head_dim=K,
        fp8_dtype=fp8_dtype,
    )

    k_q_ref, scale_ref, zero_ref, k_res_ref = ref
    k_q_fused, scale_fused, zero_fused, k_res_fused = fused

    packed_mismatch = (k_q_ref != k_q_fused).float().mean().item()
    scale_max = (scale_ref.float() - scale_fused.float()).abs().max().item()
    zero_max = (zero_ref.float() - zero_fused.float()).abs().max().item()
    res_mean = (k_res_ref.float() - k_res_fused.float()).abs().mean().item()
    if packed_mismatch > 0.01 or scale_max > 0.05 or zero_max > 0.05 or res_mean > 0.05:
        raise AssertionError(
            "fused output mismatch: "
            f"packed_mismatch={packed_mismatch:.6f}, "
            f"scale_max={scale_max:.6f}, zero_max={zero_max:.6f}, res_mean={res_mean:.6f}"
        )

    print(
        "[OK] triton fused output matches reference within tolerances "
        f"(packed_mismatch={packed_mismatch:.6f}, scale_max={scale_max:.6f}, "
        f"zero_max={zero_max:.6f}, res_mean={res_mean:.6f})"
    )
    print(
        f"device={device}, dtype={dtype}, fp8_dtype={fp8_dtype}, "
        f"B={B}, T={T}, H={H}, HKV={HKV}, K={K}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse K projection + RoPE + int2/fp8 split and test.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--HKV", type=int, default=4)
    p.add_argument("--K", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    if device.type == "cpu":
        raise RuntimeError("This test requires a CUDA device for Triton.")
    fp8_dtype = _resolve_fp8_dtype(device)
    if not hasattr(torch, "float8_e5m2") or fp8_dtype != torch.float8_e5m2:
        print("[Info] fp8 dtype not available on this device, using fp16 residuals.")

    run_smoke_test(
        device=device,
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        B=args.B,
        T=args.T,
        H=args.H,
        HKV=args.HKV,
        K=args.K,
    )


if __name__ == "__main__":
    main()

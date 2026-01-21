"""
Merge kernel for combining Q2FP8 quantized attention output with FP16 current tokens.

Uses online softmax algorithm to merge:
- o1, m1, l1: from Q2FP8 kernel (quantized part)
- o2, m2, l2: from FP16 attention (current part)

Output: o_final = merged attention output
"""

import torch
import triton
import triton.language as tl


@triton.jit
def merge_attention_kernel(
    # Q2FP8 outputs (quantized part)
    o1_ptr,  # [B, HQ, V]
    m1_ptr,  # [B, HQ] - max in log2 scale
    l1_ptr,  # [B, HQ] - sum in log2 scale
    # Query
    q_ptr,  # [B, 1, HQ, K]
    # Current FP16 K/V
    k_current_ptr,  # [B, BS, HKV, K]
    v_current_ptr,  # [B, BS, HKV, V]
    # Output
    o_final_ptr,  # [B, HQ, V]
    # Dimensions
    B,
    HQ,
    HKV,
    K,
    V,
    current_len_ptr,  # [1] int32 tensor
    scale,  # Attention scale (1/sqrt(K))
    # Strides
    stride_o1_b,
    stride_o1_h,
    stride_o1_v,
    stride_m1_b,
    stride_m1_h,
    stride_l1_b,
    stride_l1_h,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_k,
    stride_kc_b,
    stride_kc_t,
    stride_kc_h,
    stride_kc_k,
    stride_vc_b,
    stride_vc_t,
    stride_vc_h,
    stride_vc_v,
    stride_of_b,
    stride_of_h,
    stride_of_v,
    # Meta
    BLOCK_V: tl.constexpr,
    BS: tl.constexpr,
):
    """
    Merge Q2FP8 output with FP16 current tokens using online softmax.

    Modified for CUDA Graph: current_len is loaded from a tensor.
    Loop bound is fixed to BS.
    """
    # Program ID
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load current_len from tensor
    current_len = tl.load(current_len_ptr)

    # If current_len == 0, just copy o1 to output
    if current_len == 0:
        # Load o1 and write to output
        offs_v = tl.arange(0, BLOCK_V)
        for v_start in range(0, V, BLOCK_V):
            v_mask = (v_start + offs_v) < V
            o1 = tl.load(
                o1_ptr
                + pid_b * stride_o1_b
                + pid_h * stride_o1_h
                + (v_start + offs_v) * stride_o1_v,
                mask=v_mask,
                other=0.0,
            )
            tl.store(
                o_final_ptr
                + pid_b * stride_of_b
                + pid_h * stride_of_h
                + (v_start + offs_v) * stride_of_v,
                o1,
                mask=v_mask,
            )
        return

    # Load m1, l1 from Q2FP8 kernel
    m1 = tl.load(m1_ptr + pid_b * stride_m1_b + pid_h * stride_m1_h)
    l1 = tl.load(l1_ptr + pid_b * stride_l1_b + pid_h * stride_l1_h)

    # Compute which KV head this query head uses (GQA)
    num_groups = HQ // HKV
    kv_head = pid_h // num_groups

    # Load query: [K]
    offs_k = tl.arange(0, 128)  # Assume K <= 128
    k_mask = offs_k < K
    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + 0 * stride_q_t
        + pid_h * stride_q_h
        + offs_k * stride_q_k,
        mask=k_mask,
        other=0.0,
    )

    # Compute attention scores for current tokens
    # scores = q @ k_current^T * scale
    RCP_LN2 = 1.4426950408889634  # 1/ln(2) for log2 scale

    m2 = -float("inf")
    l2 = 0.0

    # Accumulator for o2 (unnormalized)
    offs_v = tl.arange(0, BLOCK_V)
    o2_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

    # Process each token in current buffer (fixed loop BS)
    for t in range(BS):
        # Mask out invalid tokens
        if t < current_len:
            # Load k_current[t]: [K]
            k_t = tl.load(
                k_current_ptr
                + pid_b * stride_kc_b
                + t * stride_kc_t
                + kv_head * stride_kc_h
                + offs_k * stride_kc_k,
                mask=k_mask,
                other=0.0,
            )

            # Compute score: q @ k_t
            score = tl.sum(q * k_t) * scale * RCP_LN2  # Convert to log2 scale

            # Update m2 (running max)
            m2_new = tl.maximum(m2, score)

            # Update l2 (running sum of exp)
            if m2 == -float("inf"):
                l2 = tl.exp2(score - m2_new)
            else:
                l2 = l2 * tl.exp2(m2 - m2_new) + tl.exp2(score - m2_new)

            m2 = m2_new

            # Load v_current[t] and accumulate to o2
            # Process V dimension in blocks
            for v_start in range(0, V, BLOCK_V):
                v_mask = (v_start + offs_v) < V
                v_t = tl.load(
                    v_current_ptr
                    + pid_b * stride_vc_b
                    + t * stride_vc_t
                    + kv_head * stride_vc_h
                    + (v_start + offs_v) * stride_vc_v,
                    mask=v_mask,
                    other=0.0,
                )

                # Accumulate: o2 += exp2(score - m2) * v_t
                weight = tl.exp2(score - m2)
                if v_start == 0:
                    o2_acc = weight * v_t
                else:
                    # For subsequent blocks, we need to load and update
                    # Optimization: Since we do V loop inside T loop, o2_acc must be handled carefully.
                    # Current impl: o2_acc is single block.
                    # Simplified: We only support V <= BLOCK_V in this path for now or assume loop structure matches.
                    # Wait, the original code had `if v_start == 0: o2_acc = ...`
                    # But o2_acc is declared as `zeros`.
                    # Correct logic for V-tiling inside T-loop is tricky if V > BLOCK_V.
                    # But here V=128 usually.
                    pass

    # Re-loop for O2 accumulation with correct m2 (softmax)
    # Since we can't easily store all intermediate scores, we might need 2 passes or just recompute.
    # The original kernel did recompute.

    # PASS 2: Accumulate o2
    # Reset o2 accumulators (we need one per V-block)
    # But since we have V-loop outside in merge part, let's look at original structure.
    # Original:
    # m2/l2 pass
    # then merge pass (V loop):
    #   recompute o2 for this V-block
    #   merge

    m_new = tl.maximum(m1, m2)
    alpha1 = tl.exp2(m1 - m_new)
    alpha2 = tl.exp2(m2 - m_new)
    l_new = l1 * alpha1 + l2 * alpha2

    # Process V dimension in blocks
    for v_start in range(0, V, BLOCK_V):
        v_mask = (v_start + offs_v) < V

        # Load o1
        o1 = tl.load(
            o1_ptr
            + pid_b * stride_o1_b
            + pid_h * stride_o1_h
            + (v_start + offs_v) * stride_o1_v,
            mask=v_mask,
            other=0.0,
        )

        # Recompute o2 for this V block
        o2 = tl.zeros([BLOCK_V], dtype=tl.float32)

        # Loop T again
        for t in range(BS):
            if t < current_len:
                k_t = tl.load(
                    k_current_ptr
                    + pid_b * stride_kc_b
                    + t * stride_kc_t
                    + kv_head * stride_kc_h
                    + offs_k * stride_kc_k,
                    mask=k_mask,
                    other=0.0,
                )
                score = tl.sum(q * k_t) * scale * RCP_LN2

                v_t = tl.load(
                    v_current_ptr
                    + pid_b * stride_vc_b
                    + t * stride_vc_t
                    + kv_head * stride_vc_h
                    + (v_start + offs_v) * stride_vc_v,
                    mask=v_mask,
                    other=0.0,
                )
                o2 += tl.exp2(score - m2) * v_t

        # Merge: o_final = (o1 * l1 * alpha1 + o2 * alpha2) / l_new
        o_final = (o1 * l1 * alpha1 + o2 * alpha2) / l_new

        # Store result
        tl.store(
            o_final_ptr
            + pid_b * stride_of_b
            + pid_h * stride_of_h
            + (v_start + offs_v) * stride_of_v,
            o_final,
            mask=v_mask,
        )


def merge_attention_output(
    o1: torch.Tensor,  # [B, HQ, V] - Q2FP8 output
    m1: torch.Tensor,  # [B, HQ] - max from Q2FP8
    l1: torch.Tensor,  # [B, HQ] - sum from Q2FP8
    q: torch.Tensor,  # [B, 1, HQ, K] - query
    k_current: torch.Tensor,  # [B, BS, HKV, K] - current FP16 keys
    v_current: torch.Tensor,  # [B, BS, HKV, V] - current FP16 values
    current_len: torch.Tensor | int,  # Tensor[1] or int
    scale: float = None,  # Attention scale
    o_final: torch.Tensor = None,  # Pre-allocated output buffer (optional)
) -> torch.Tensor:
    """
    Merge Q2FP8 attention output with FP16 current tokens.
    """
    B, HQ, V = o1.shape
    _, _, _, K = q.shape
    _, BS, HKV, _ = k_current.shape

    if scale is None:
        scale = 1.0 / (K**0.5)

    if o_final is None:
        o_final = torch.empty_like(o1)

    # Handle current_len: ensure it's a tensor on the same device
    if isinstance(current_len, int):
        current_len_tensor = torch.tensor(
            [current_len], device=o1.device, dtype=torch.int32
        )
    elif isinstance(current_len, torch.Tensor):
        if current_len.ndim == 0:
            current_len_tensor = current_len.view(1)
        else:
            current_len_tensor = current_len
        if current_len_tensor.device != o1.device:
            current_len_tensor = current_len_tensor.to(o1.device)
    else:
        raise ValueError("current_len must be int or Tensor")

    # Launch kernel
    grid = (B, HQ)
    BLOCK_V = 128

    merge_attention_kernel[grid](
        o1,
        m1,
        l1,
        q,
        k_current,
        v_current,
        o_final,
        B,
        HQ,
        HKV,
        K,
        V,
        current_len_tensor,
        scale,
        # Strides for o1
        o1.stride(0),
        o1.stride(1),
        o1.stride(2),
        # Strides for m1
        m1.stride(0),
        m1.stride(1),
        # Strides for l1
        l1.stride(0),
        l1.stride(1),
        # Strides for q
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # Strides for k_current
        k_current.stride(0),
        k_current.stride(1),
        k_current.stride(2),
        k_current.stride(3),
        # Strides for v_current
        v_current.stride(0),
        v_current.stride(1),
        v_current.stride(2),
        v_current.stride(3),
        # Strides for o_final
        o_final.stride(0),
        o_final.stride(1),
        o_final.stride(2),
        # Meta
        BLOCK_V=BLOCK_V,
        BS=BS,  # Passed as constexpr
    )

    return o_final

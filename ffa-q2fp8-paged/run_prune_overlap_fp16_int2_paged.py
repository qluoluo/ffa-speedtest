# Compare fp16 vs int2 pruning overlap following paged attention logic.
import argparse
import math
import re
import sys
from pathlib import Path

import torch

from utils.load import load_qkvh

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

EXP_ROOT_DIR = Path(__file__).resolve().parents[1] / "attn_analysis" / "result"
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")

RCP_LN2 = 1.4426950408889634
QZERO = 1.5
QMAX = 3.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare fp16/int2 pruning overlap using paged attention threshold logic."
    )
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--SBS", type=int, default=256, help="Page size (tokens per page).")
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--t-bs", type=int, default=16, help="Tokens per page used for threshold.")
    p.add_argument("--exp-root", type=str, default=None)
    p.add_argument("--exp-subdir", type=str, default=None)
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def convert_layout(q_rope_1: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor):
    B, Hq, qlen, Dq = q_rope_1.shape
    if k_rope.dim() != 4 or v.dim() != 4:
        raise ValueError("k_rope and v must be 4D tensors")
    Bk, dim1, dim2, Dk = k_rope.shape
    Bv, dim1v, dim2v, Dv = v.shape
    if B != Bk or B != Bv:
        raise ValueError("Batch size mismatch in q/k/v")

    q = q_rope_1[:, :, -1, :].contiguous()

    if dim1 == dim1v and dim2 == dim2v and dim1 == dim1v:
        k = k_rope.permute(0, 2, 1, 3).contiguous()
        v_out = v.permute(0, 2, 1, 3).contiguous()
    elif dim2 == dim2v:
        k = k_rope.contiguous()
        v_out = v.contiguous()
    else:
        raise ValueError("k_rope/v layouts must be [B, Hkv, T, D] or [B, T, Hkv, D]")

    return q, k, v_out


def load_layer_batch(layer_data_root, layer_indices, dtype, max_length):
    layer_qkvh_data_list = []
    data_iter = load_qkvh(
        layer_data_root, device="cuda", start_layer=layer_indices[0], max_length=max_length
    )
    for i, layer_idx in enumerate(layer_indices):
        try:
            layer_data = next(data_iter)
        except StopIteration:
            raise RuntimeError(
                f"Not enough layers to form batch size {len(layer_indices)} starting from layer_{layer_indices[0]}. "
                f"Only found {i} layers."
            )
        layer_qkvh_data_list.append(layer_data)
        print(f"[Info] Loaded data for layer_{layer_idx}")

    q_rope_list = [layer_data["q_rope"] for layer_data in layer_qkvh_data_list]
    k_rope_list = [layer_data["k_rope"] for layer_data in layer_qkvh_data_list]
    v_list = [layer_data["v"] for layer_data in layer_qkvh_data_list]

    q_rope_full = torch.cat(q_rope_list, dim=0).to(dtype=dtype)
    k_rope_full = torch.cat(k_rope_list, dim=0).to(dtype=dtype)
    v_full = torch.cat(v_list, dim=0).to(dtype=dtype)

    return q_rope_full, k_rope_full, v_full


def quantize_k_2bit_symmetric(k: torch.Tensor):
    k_absmax = k.abs().amax(dim=1)
    scale = (k_absmax / QZERO).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + QZERO).clamp(0, QMAX).to(torch.uint8)
    return k_q, scale


def build_pages(x: torch.Tensor, page_size: int):
    B, T, HKV, K = x.shape
    n_pages = math.ceil(T / page_size)
    pad = n_pages * page_size - T
    if pad:
        pad_tensor = torch.zeros((B, pad, HKV, K), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=1)

    pages = x.view(B, n_pages, page_size, HKV, K).permute(0, 1, 3, 2, 4).contiguous()
    page_lens = torch.full((B, n_pages), page_size, device=x.device, dtype=torch.int32)
    if pad:
        page_lens[:, -1] = page_size - pad
    return pages, page_lens


def compute_threshold_int2(
    q: torch.Tensor,
    k_pages_q: torch.Tensor,
    k_scale: torch.Tensor,
    page_lens: torch.Tensor,
    *,
    delta: float,
    t_bs: int,
):
    B, HQ, K = q.shape
    _, n_pages, HKV, _, _ = k_pages_q.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")
    G = HQ // HKV
    softmax_scale = 1.0 / math.sqrt(K)
    neg_inf = float("-inf")

    q_f = q.float()
    k_scale_f = k_scale.float()
    q_group = q_f.view(B, HKV, G, K)
    q_scaled = q_group * k_scale_f[:, :, None, :]
    q_zero_sum = -QZERO * q_scaled.sum(dim=-1)

    threshold = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
    for b in range(B):
        len0 = int(page_lens[b, 0].item())
        len1 = int(page_lens[b, -1].item())
        t0 = min(t_bs, len0)
        t1 = min(t_bs, len1)
        for h in range(HKV):
            q_scaled_h = q_scaled[b, h]
            q_zero_h = q_zero_sum[b, h]
            k0 = k_pages_q[b, 0, h, :t_bs, :].float()
            k1 = k_pages_q[b, -1, h, :t_bs, :].float()
            logits0 = torch.matmul(q_scaled_h, k0.T) + q_zero_h[:, None]
            logits1 = torch.matmul(q_scaled_h, k1.T) + q_zero_h[:, None]
            if t0 < t_bs:
                logits0[:, t0:] = neg_inf
            if t1 < t_bs:
                logits1[:, t1:] = neg_inf
            logits0 = logits0 * softmax_scale * RCP_LN2
            logits1 = logits1 * softmax_scale * RCP_LN2
            m0 = logits0.max(dim=1).values
            m1 = logits1.max(dim=1).values
            th = torch.maximum(m0, m1) - float(delta)
            threshold[b, h * G : (h + 1) * G] = th
    return threshold


def compute_threshold_fp16(
    q: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    page_lens: torch.Tensor,
    *,
    delta: float,
    t_bs: int,
):
    B, HQ, K = q.shape
    _, n_pages, HKV, _, _ = k_pages_fp16.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")
    G = HQ // HKV
    softmax_scale = 1.0 / math.sqrt(K)
    neg_inf = float("-inf")

    q_f = q.float()
    q_group = q_f.view(B, HKV, G, K)

    threshold = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
    for b in range(B):
        len0 = int(page_lens[b, 0].item())
        len1 = int(page_lens[b, -1].item())
        t0 = min(t_bs, len0)
        t1 = min(t_bs, len1)
        for h in range(HKV):
            q_h = q_group[b, h]
            k0 = k_pages_fp16[b, 0, h, :t_bs, :].float()
            k1 = k_pages_fp16[b, -1, h, :t_bs, :].float()
            logits0 = torch.matmul(q_h, k0.T)
            logits1 = torch.matmul(q_h, k1.T)
            if t0 < t_bs:
                logits0[:, t0:] = neg_inf
            if t1 < t_bs:
                logits1[:, t1:] = neg_inf
            logits0 = logits0 * softmax_scale * RCP_LN2
            logits1 = logits1 * softmax_scale * RCP_LN2
            m0 = logits0.max(dim=1).values
            m1 = logits1.max(dim=1).values
            th = torch.maximum(m0, m1) - float(delta)
            threshold[b, h * G : (h + 1) * G] = th
    return threshold


def prune_mask_int2(
    q: torch.Tensor,
    k_pages_q: torch.Tensor,
    k_scale: torch.Tensor,
    page_lens: torch.Tensor,
    threshold: torch.Tensor,
):
    B, HQ, K = q.shape
    _, n_pages, HKV, SBS, _ = k_pages_q.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")
    G = HQ // HKV
    softmax_scale = 1.0 / math.sqrt(K)
    neg_inf = float("-inf")

    q_f = q.float()
    k_scale_f = k_scale.float()
    q_group = q_f.view(B, HKV, G, K)
    q_scaled = q_group * k_scale_f[:, :, None, :]
    q_zero_sum = -QZERO * q_scaled.sum(dim=-1)

    mask = torch.zeros((B, HKV, n_pages), device=q.device, dtype=torch.bool)
    token_idx = torch.arange(SBS, device=q.device)
    for b in range(B):
        page_lens_b = page_lens[b]
        page_mask = token_idx[None, :] >= page_lens_b[:, None]
        for h in range(HKV):
            q_scaled_h = q_scaled[b, h]
            q_zero_h = q_zero_sum[b, h]
            k_pages_h = k_pages_q[b, :, h, :, :]
            k_mat = k_pages_h.reshape(n_pages * SBS, K).float()
            logits = torch.matmul(q_scaled_h, k_mat.T) + q_zero_h[:, None]
            logits = logits.view(G, n_pages, SBS).permute(1, 0, 2)
            logits = logits * softmax_scale * RCP_LN2
            if page_mask.any():
                logits = logits.masked_fill(page_mask[:, None, :], neg_inf)
            m_rows = logits.max(dim=2).values
            th_rows = threshold[b, h * G : (h + 1) * G]
            mask[b, h] = (m_rows < th_rows[None, :]).all(dim=1)
    return mask


def prune_mask_fp16(
    q: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    page_lens: torch.Tensor,
    threshold: torch.Tensor,
):
    B, HQ, K = q.shape
    _, n_pages, HKV, SBS, _ = k_pages_fp16.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")
    G = HQ // HKV
    softmax_scale = 1.0 / math.sqrt(K)
    neg_inf = float("-inf")

    q_f = q.float()
    q_group = q_f.view(B, HKV, G, K)

    mask = torch.zeros((B, HKV, n_pages), device=q.device, dtype=torch.bool)
    token_idx = torch.arange(SBS, device=q.device)
    for b in range(B):
        page_lens_b = page_lens[b]
        page_mask = token_idx[None, :] >= page_lens_b[:, None]
        for h in range(HKV):
            q_h = q_group[b, h]
            k_pages_h = k_pages_fp16[b, :, h, :, :]
            k_mat = k_pages_h.reshape(n_pages * SBS, K).float()
            logits = torch.matmul(q_h, k_mat.T)
            logits = logits.view(G, n_pages, SBS).permute(1, 0, 2)
            logits = logits * softmax_scale * RCP_LN2
            if page_mask.any():
                logits = logits.masked_fill(page_mask[:, None, :], neg_inf)
            m_rows = logits.max(dim=2).values
            th_rows = threshold[b, h * G : (h + 1) * G]
            mask[b, h] = (m_rows < th_rows[None, :]).all(dim=1)
    return mask


def summarize_prune(mask: torch.Tensor):
    total = mask.numel()
    pruned = int(mask.sum().item())
    ratio = float(pruned / total) if total > 0 else 0.0
    return pruned, total, ratio


def overlap_stats(a: torch.Tensor, b: torch.Tensor):
    inter = int((a & b).sum().item())
    union = int((a | b).sum().item())
    a_sum = int(a.sum().item())
    b_sum = int(b.sum().item())
    return {
        "inter": inter,
        "union": union,
        "a_sum": a_sum,
        "b_sum": b_sum,
        "jaccard": inter / union if union > 0 else 0.0,
        "a_recall": inter / a_sum if a_sum > 0 else 0.0,
        "b_recall": inter / b_sum if b_sum > 0 else 0.0,
    }


def get_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this test.")
    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = props.name.strip()
    total_mem_gb = math.ceil(props.total_memory / (1024**3))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "gpu"
    tag = f"{safe_name}_{total_mem_gb}GB"
    return tag, name, total_mem_gb, device_idx


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    SBS = int(args.SBS)
    delta = float(args.delta)
    t_bs = int(args.t_bs)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    _, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    exp_root = Path(args.exp_root) if args.exp_root else EXP_ROOT_DIR
    exp_subdir = Path(args.exp_subdir) if args.exp_subdir else EXP_ROOT_SUBDIR
    exp_root = exp_root / exp_subdir
    layer_data_root = exp_root / "layer_data"

    layer_indices = list(range(args.layer, args.layer + bsz))
    q_rope_full, k_rope_full, v_full = load_layer_batch(
        layer_data_root, layer_indices, dtype, max_length
    )

    B, HQ, T_full, K = q_rope_full.shape
    L = args.length if args.length is not None else T_full
    if L <= 0 or L > T_full:
        raise ValueError(f"--length must be in [1, {T_full}], got {L}")

    q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
    k_rope = k_rope_full[:, :, :L, :].contiguous()
    v = v_full[:, :, :L, :].contiguous()
    q, k, _ = convert_layout(q_rope_1, k_rope, v)

    _, _, HKV, _ = k.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ must be divisible by HKV: {HQ=} {HKV=}")

    print(f"[Info] Layers={layer_indices}, bsz={B}, Hq={HQ}, Hkv={HKV}, T={L}, K={K}")

    k_q_unpacked, k_scale = quantize_k_2bit_symmetric(k)
    k_pages_q, page_lens = build_pages(k_q_unpacked, SBS)
    k_pages_fp16, _ = build_pages(k, SBS)

    threshold_int2 = compute_threshold_int2(
        q, k_pages_q, k_scale, page_lens, delta=delta, t_bs=t_bs
    )
    threshold_fp16 = compute_threshold_fp16(
        q, k_pages_fp16, page_lens, delta=delta, t_bs=t_bs
    )

    mask_int2 = prune_mask_int2(q, k_pages_q, k_scale, page_lens, threshold_int2)
    mask_fp16 = prune_mask_fp16(q, k_pages_fp16, page_lens, threshold_fp16)

    pruned_int2, total_blocks, ratio_int2 = summarize_prune(mask_int2)
    pruned_fp16, _, ratio_fp16 = summarize_prune(mask_fp16)
    overlap = overlap_stats(mask_fp16, mask_int2)

    print(
        f"[Result] int2_pruned={pruned_int2}/{total_blocks} (ratio={ratio_int2:.4f})"
    )
    print(
        f"[Result] fp16_pruned={pruned_fp16}/{total_blocks} (ratio={ratio_fp16:.4f})"
    )
    print(
        "[Result] overlap "
        f"inter={overlap['inter']} jaccard={overlap['jaccard']:.4f} "
        f"fp16_recall={overlap['a_recall']:.4f} int2_recall={overlap['b_recall']:.4f}"
    )


if __name__ == "__main__":
    main()

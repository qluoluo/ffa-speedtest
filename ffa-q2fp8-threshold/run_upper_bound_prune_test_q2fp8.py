# Test upper-bound pruning for Q2FP8 decode.
import argparse
import math
import re
import sys
from pathlib import Path

import torch

from utils.load import load_qkvh

# Ensure package importability
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import (
    attn_compute_threshold_qbits,
    attn_forward_decode_quantized,
)

EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")

RCP_LN2 = 1.4426950408889634


def parse_args():
    p = argparse.ArgumentParser(description="Upper-bound prune test for Q2FP8 decode.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=256)
    p.add_argument("--SBS", type=int, default=None)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--length", type=int, default=None, help="Test a single length (<= T_full).")
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def convert_layout(q_rope_1: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor):
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv
    q = q_rope_1[:, :, 0, :].contiguous()
    k = k_rope.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def quantize_k_2bit_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2):
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    values_per_byte = 4
    B, T, HKV, K = k_q.shape
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


def compute_block_max(values: torch.Tensor, BS: int, SBS: int) -> torch.Tensor:
    if BS % SBS != 0:
        raise ValueError(f"BS must be divisible by SBS for this test, got BS={BS}, SBS={SBS}")
    B, T, HKV = values.shape
    NTB = math.ceil(T / BS)
    NSB = BS // SBS
    total = NTB * BS
    if total > T:
        pad = total - T
        pad_tensor = torch.zeros((B, pad, HKV), device=values.device, dtype=values.dtype)
        values = torch.cat([values, pad_tensor], dim=1)
    values = values.view(B, NTB, BS, HKV)
    values = values.view(B, NTB, NSB, SBS, HKV)
    return values.max(dim=3).values  # [B, NTB, NSB, HKV]


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    _, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))

    q_rope_full_list, k_rope_full_list, v_full_list = [], [], []
    data_iter = load_qkvh(layer_data_root, device="cuda", start_layer=layer_indices[0], max_length=max_length)
    for i, layer_idx in enumerate(layer_indices):
        try:
            layer_data = next(data_iter)
        except StopIteration:
            raise RuntimeError(
                f"Not enough layers to form batch size {len(layer_indices)} starting from layer_{layer_indices[0]}. "
                f"Only found {i} layers."
            )
        q_rope_full_list.append(layer_data["q_rope"])
        k_rope_full_list.append(layer_data["k_rope"])
        v_full_list.append(layer_data["v"])
        print(f"[Info] Loaded data for layer_{layer_idx}")

    q_rope_full = torch.cat(q_rope_full_list, dim=0).to(dtype=dtype)
    k_rope_full = torch.cat(k_rope_full_list, dim=0).to(dtype=dtype)
    v_full = torch.cat(v_full_list, dim=0).to(dtype=dtype)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)
    G = Hq // Hkv

    L = args.length if args.length is not None else T_full
    if L <= 0 or L > T_full:
        raise ValueError(f"--length must be in [1, {T_full}], got {L}")

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T={L}, K={K}, V={V}")

    q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
    k_rope = k_rope_full[:, :, :L, :].contiguous()
    v = v_full[:, :, :L, :].contiguous()

    q, k, v = convert_layout(q_rope_1, k_rope, v)
    q_1 = q.unsqueeze(1)

    k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)

    B, _, HQ, D = q_1.shape
    T = k_q.shape[1]
    K_packed = k_q.shape[3]
    NTB = math.ceil(T / BS)

    threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
    attn_compute_threshold_qbits[(B, Hkv)](
        q, k_q, k_scale, k_zero,
        threshold_buf,
        scale, T, NTB, delta,
        B=B, HKV=Hkv, HQ=HQ, K=D, K_PACKED=K_packed, G=G,
        K_BITS=2,
    )

    _, actual_skip_ratio = attn_forward_decode_quantized(
        q=q_1,
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        k_residual=k_residual,
        v=v,
        k_bits=2,
        scale=scale,
        BS=BS,
        SBS=SBS,
        delta=delta,
        return_skip_ratio=True,
        use_fp8_residual=True,
    )

    q_f = q.float()
    q_l2 = torch.sqrt(torch.sum(q_f * q_f, dim=-1))  # [B, HQ]
    q_l1 = torch.sum(q_f.abs(), dim=-1)  # [B, HQ]

    k_f = k.float()
    k_l2 = torch.sqrt(torch.sum(k_f * k_f, dim=-1))  # [B, T, HKV]
    k_abs = k_f.abs().amax(dim=-1)  # [B, T, HKV]

    k_l2_blk = compute_block_max(k_l2, BS, SBS).permute(0, 3, 1, 2)  # [B, HKV, NTB, NSB]
    k_abs_blk = compute_block_max(k_abs, BS, SBS).permute(0, 3, 1, 2)  # [B, HKV, NTB, NSB]

    q_l2_group = q_l2.view(B, Hkv, G)[:, :, :, None, None]
    q_l1_group = q_l1.view(B, Hkv, G)[:, :, :, None, None]
    th_group = threshold_buf.view(B, Hkv, G)[:, :, :, None, None]

    bound_l2 = q_l2_group * k_l2_blk[:, :, None, :, :] * scale * RCP_LN2
    bound_abs = q_l1_group * k_abs_blk[:, :, None, :, :] * scale * RCP_LN2

    prune_l2 = (bound_l2 < th_group).all(dim=2)
    prune_abs = (bound_abs < th_group).all(dim=2)

    total_blocks = prune_l2.numel()
    prune_l2_ratio = float(prune_l2.sum().item() / total_blocks)
    prune_abs_ratio = float(prune_abs.sum().item() / total_blocks)

    coverage_l2 = prune_l2_ratio / actual_skip_ratio if actual_skip_ratio > 0 else 0.0
    coverage_abs = prune_abs_ratio / actual_skip_ratio if actual_skip_ratio > 0 else 0.0

    print(f"[Result] actual_skip_ratio={actual_skip_ratio:.4f}")
    print(
        f"[Result] bound_l2_prune_ratio={prune_l2_ratio:.4f} "
        f"(coverage_vs_actual={coverage_l2:.3f})"
    )
    print(
        f"[Result] bound_abs_prune_ratio={prune_abs_ratio:.4f} "
        f"(coverage_vs_actual={coverage_abs:.3f})"
    )


if __name__ == "__main__":
    main()

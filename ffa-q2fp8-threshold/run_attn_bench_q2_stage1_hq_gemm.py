# Prototype benchmark: merged-HQ GEMM for QK block maxima + threshold.
import argparse
import math
import re
import sys
from pathlib import Path

import torch

from utils.bench import benchmark
from utils.load import load_qkvh

from attn_kernel.attn_kernel_v1210_stage1_hq_gemm import (
    qk_block_max_hq_gemm_forward,
    threshold_from_blocks,
)

# Ensure package importability
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark merged-HQ GEMM QK+threshold prototype.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=256)
    p.add_argument("--SBS", type=int, default=256)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--length", type=int, default=None, help="Benchmark a single length (<= T_full).")
    p.add_argument("--iters", type=int, default=200, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup")
    p.add_argument("--num-warps", type=int, default=4, help="Triton num_warps for prototype kernel.")
    p.add_argument("--num-stages", type=int, default=2, help="Triton num_stages for prototype kernel.")
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


def quantize_k_2bit(k: torch.Tensor):
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)

    values_per_byte = 4  # 8 bits / 2 bits
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
    return k_q_packed, scale, zero


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


def get_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

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
    BS = int(args.BS)
    SBS = int(args.SBS)
    delta = float(args.delta)
    iters = int(args.iters)
    warmup = int(args.warmup)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    if args.length is not None:
        L = int(args.length)
        if L <= 0 or L > T_full:
            raise ValueError(f"Invalid length {L}, expected 1..{T_full}")
    else:
        L = T_full

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T={L}, K={K}, V={V}")

    q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
    k_rope = k_rope_full[:, :, :L, :].contiguous()
    v = v_full[:, :, :L, :].contiguous()

    q, k, _ = convert_layout(q_rope_1, k_rope, v)
    q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
    k_q, k_scale, k_zero = quantize_k_2bit(k)

    def run_qk():
        m_buf = qk_block_max_hq_gemm_forward(
            q_1,
            k_q,
            k_scale,
            k_zero,
            scale=scale,
            SBS=SBS,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        _ = threshold_from_blocks(m_buf, delta)
        return m_buf

    ms_qk = benchmark(run_qk, iters=iters, warmup=warmup)
    print(
        f"[Result] QK+TH merged-HQ GEMM | T={L} | BS={BS} SBS={SBS} delta={delta} | "
        f"QK_TH={ms_qk:.3f} ms (warps={args.num_warps}, stages={args.num_stages})"
    )


if __name__ == "__main__":
    main()

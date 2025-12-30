# Benchmark Q2FP8 decode with CUDAGraph and save a report file with GPU/run info.
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from utils.bench import benchmark
from utils.cache import dtype_key, to_k_str
from utils.load import load_qkvh

from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import attn_forward_decode_quantized
from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_cudagraph import (
    CUDAGraphDecodeRunnerQ2FP8,
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
    p = argparse.ArgumentParser(description="Benchmark Q2FP8 decode (CUDAGraph) and write a report JSON.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128)
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
    p.add_argument("--length", type=int, default=None, help="Benchmark a single length (<= T_full).")
    p.add_argument("--step", type=int, default=1024, help="Step size for length sweep.")
    p.add_argument("--iters", type=int, default=500, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=100, help="Benchmark warmup")
    p.add_argument("--cg-warmup", type=int, default=2, help="CUDAGraph warmup calls before capture")
    p.add_argument(
        "--cg-replay-only",
        action="store_true",
        default=True,
        help="Measure CUDAGraph replay time only (exclude input copies).",
    )
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output report JSON path (default: auto under plot/).",
    )
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
    return k_q_packed, scale, zero, k_residual


def get_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = props.name.strip()
    total_mem_gb = math.ceil(props.total_memory / (1024**3))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "gpu"
    tag = f"{safe_name}_{total_mem_gb}GB"
    return tag, props, device_idx


def build_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, base_dir: Path):
    layer_range = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"
    lmax_name = str(max_length) if max_length is not None else ""
    root_dir = (
        base_dir
        / "plot"
        / f"{attn_kernel_name}_cudagraph_report"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}" + (f"_{lmax_name}" if max_length is not None else ""))
    )
    raw_data_dir = root_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return root_dir, raw_data_dir


def make_cache_file_path(
    raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1, replay_only=False
):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    raw_dir = Path(raw_data_dir)
    suffix = "_cudagraph_replay" if replay_only else "_cudagraph"
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}{suffix}.json"
    )
    return raw_dir / fname


def save_raw_cache(path, meta: dict, lengths, q2_ms, q2_cg_ms, flash_ms, skip_ratios):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "q2_ms": [float(x) for x in q2_ms],
        "q2_cg_ms": [float(x) for x in q2_cg_ms],
        "flash_ms": [None if x is None else float(x) for x in flash_ms],
        "skip_ratios": [None if x is None else float(x) for x in skip_ratios],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_raw_cache(path):
    with Path(path).open("r") as f:
        data = json.load(f)
    return (
        data["lengths"],
        data["q2_ms"],
        data["q2_cg_ms"],
        data.get("flash_ms", [None] * len(data["lengths"])),
        data.get("skip_ratios", [None] * len(data["lengths"])),
        data.get("meta", {}),
    )


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


def maybe_load_flash(no_flash: bool):
    if no_flash:
        return None, "disabled"
    try:
        from utils.flash import flash_attn_compute
        return flash_attn_compute, None
    except Exception as exc:
        return None, str(exc)


def collect_software_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    try:
        import triton  # type: ignore
        info["triton_version"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        info["triton_version"] = f"unavailable: {exc}"
    try:
        import flash_attn  # type: ignore
        info["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:
        info["flash_attn_version"] = f"unavailable: {exc}"
    return info


def build_report(
    args,
    gpu_props,
    device_idx,
    gpu_tag,
    lengths,
    q2_ms_list,
    q2_cg_ms_list,
    flash_ms_list,
    skip_ratios,
    meta,
):
    def _speedup(base, faster):
        if base is None or faster is None or faster <= 0:
            return None
        return float(base) / float(faster)

    speedup = [_speedup(f, q) for f, q in zip(flash_ms_list, q2_ms_list)]
    speedup_cg = [_speedup(f, q) for f, q in zip(flash_ms_list, q2_cg_ms_list)]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _median(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals = sorted(vals)
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    gpu_info = {
        "device_index": device_idx,
        "name": gpu_props.name.strip(),
        "total_memory_gb": math.ceil(gpu_props.total_memory / (1024**3)),
        "sm_count": gpu_props.multi_processor_count,
        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
        "clock_rate_khz": getattr(gpu_props, "clock_rate", None),
        "memory_clock_rate_khz": getattr(gpu_props, "memory_clock_rate", None),
        "memory_bus_width": getattr(gpu_props, "memory_bus_width", None),
    }

    report = {
        "report_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cwd": str(Path.cwd()),
        "cmdline": " ".join(sys.argv),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": gpu_info,
        "software": collect_software_info(),
        "run_config": vars(args),
        "data_meta": meta,
        "gpu_tag": gpu_tag,
        "lengths": lengths,
        "q2_ms": q2_ms_list,
        "q2_cg_ms": q2_cg_ms_list,
        "flash_ms": flash_ms_list,
        "skip_ratios": skip_ratios,
        "speedup": speedup,
        "speedup_cg": speedup_cg,
        "summary": {
            "length_last": lengths[-1] if lengths else None,
            "q2_ms_last": q2_ms_list[-1] if q2_ms_list else None,
            "q2_cg_ms_last": q2_cg_ms_list[-1] if q2_cg_ms_list else None,
            "flash_ms_last": flash_ms_list[-1] if flash_ms_list else None,
            "speedup_last": speedup[-1] if speedup else None,
            "speedup_cg_last": speedup_cg[-1] if speedup_cg else None,
            "speedup_mean": _mean(speedup),
            "speedup_cg_mean": _mean(speedup_cg),
            "speedup_median": _median(speedup),
            "speedup_cg_median": _median(speedup_cg),
        },
    }
    return report


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    step = int(args.step)
    iters = int(args.iters)
    warmup = int(args.warmup)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    attn_kernel_name = "attn_kernel_v1210_fused_bsz_q2fp8"

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))
    layer_range_str = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"

    gpu_tag, gpu_props, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_props.name} ({math.ceil(gpu_props.total_memory / (1024**3))}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    if args.length is not None:
        if args.length <= 0 or args.length > T_full:
            raise ValueError(f"--length must be in (0, {T_full}], got {args.length}")
        lengths = [int(args.length)]
    else:
        lengths = list(range(step, T_full, step)) + [T_full]

    flash_attn_compute, flash_err = maybe_load_flash(args.no_flash)
    if flash_attn_compute is None:
        print(f"[Info] FlashAttention baseline disabled: {flash_err}")

    report_root_dir, raw_data_dir = build_dirs(
        attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, THIS_DIR
    )
    cache_path = make_cache_file_path(
        raw_data_dir,
        f"layers_{layer_range_str}",
        T_full,
        Hq,
        Hkv,
        K,
        V,
        BS,
        SBS,
        delta,
        dtype,
        step,
        iters,
        warmup,
        bsz=bsz,
        replay_only=args.cg_replay_only,
    )

    if cache_path.exists() and not args.force:
        x_lengths, q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios, meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios = [], [], [], []
        for L in lengths:
            q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
            k_rope = k_rope_full[:, :, :L, :].contiguous()
            v = v_full[:, :, :L, :].contiguous()

            q, k, v = convert_layout(q_rope_1, k_rope, v)
            q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
            k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)

            _, skip_ratio = attn_forward_decode_quantized(
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
            )

            runner = CUDAGraphDecodeRunnerQ2FP8(
                q_1,
                k_q,
                k_scale,
                k_zero,
                v,
                k_residual=k_residual,
                k_bits=2,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
                warmup=args.cg_warmup,
            )

            def run_q2():
                return attn_forward_decode_quantized(
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
                    return_skip_ratio=False,
                )

            def run_q2_cg():
                if args.cg_replay_only:
                    return runner.replay_only()
                return runner(
                    q_1,
                    k_q,
                    k_scale,
                    k_zero,
                    v,
                    k_residual=k_residual,
                )

            def run_flash():
                return flash_attn_compute(q, k, v)

            ms_q2 = benchmark(run_q2, iters=iters, warmup=warmup)
            ms_q2_cg = benchmark(run_q2_cg, iters=iters, warmup=warmup)
            ms_flash = None
            if flash_attn_compute is not None:
                ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

            q2_ms_list.append(ms_q2)
            q2_cg_ms_list.append(ms_q2_cg)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(float(skip_ratio))

        x_lengths = lengths
        meta = dict(
            layer_indices=layer_indices,
            T_full=int(T_full),
            Hq=int(Hq),
            Hkv=int(Hkv),
            D=int(K),
            Dv=int(V),
            BS=int(BS),
            SBS=int(SBS),
            delta=float(delta),
            dtype=dtype_key(dtype),
            step=int(step),
            iters=int(iters),
            warmup=int(warmup),
            attn_kernel=attn_kernel_name,
            bsz=int(bsz),
            cudagraph=True,
            cudagraph_replay_only=bool(args.cg_replay_only),
        )
        save_raw_cache(cache_path, meta, x_lengths, q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios)
        print(f"[Info] Saved raw benchmark data to {cache_path}")

    report = build_report(
        args,
        gpu_props,
        gpu_idx,
        gpu_tag,
        x_lengths,
        q2_ms_list,
        q2_cg_ms_list,
        flash_ms_list,
        skip_ratios,
        meta,
    )

    if args.out:
        report_path = Path(args.out)
    else:
        report_root_dir.mkdir(parents=True, exist_ok=True)
        Tmax_k_str = to_k_str(meta.get("T_full", 0))
        report_name = f"layer_layers_{layer_range_str}_Tmax{Tmax_k_str}_report.json"
        report_path = report_root_dir / report_name

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[Result] Saved report to: {report_path}")


if __name__ == "__main__":
    main()

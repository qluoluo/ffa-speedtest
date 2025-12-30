# CUDAGraph stage timing with report output (threshold/stage1/stage2/full).
import argparse
import importlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import dtype_key, to_k_str
from utils.load import load_qkvh

# Ensure package importability
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="CUDAGraph stage timing for Q2FP8 decode kernels with report JSON.")
    p.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8",
        help="Python module path for Q2FP8 kernels.",
    )
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
    p.add_argument("--iters", type=int, default=200, help="Benchmark iters per stage")
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup per stage")
    p.add_argument("--cg-warmup", type=int, default=2, help="CUDAGraph warmup calls before capture")
    p.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    p.add_argument("--no-fp8-residual", action="store_true", help="Disable fp8 residual refinement.")
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


def load_kernel_components(kernel_path: str):
    kernel_module = importlib.import_module(kernel_path)
    required = [
        "attn_forward_decode_quantized",
        "attn_compute_threshold_qbits",
        "attn_forward_stage1_fused_threshold_qbits",
        "attn_forward_stage2_masked",
    ]
    missing = [name for name in required if not hasattr(kernel_module, name)]
    if missing:
        raise AttributeError(f"Module {kernel_path} missing kernels: {missing}")
    return kernel_module


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


def build_output_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, base_dir: Path):
    layer_range = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"
    lmax_name = str(max_length) if max_length is not None else ""
    root_dir = (
        base_dir
        / "plot"
        / f"{attn_kernel_name}_stage_timing_cudagraph"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}" + (f"_{lmax_name}" if max_length is not None else ""))
    )
    raw_data_dir = root_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return root_dir, raw_data_dir


def build_report_root(attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, base_dir: Path):
    layer_range = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"
    lmax_name = str(max_length) if max_length is not None else ""
    root_dir = (
        base_dir
        / "plot"
        / f"{attn_kernel_name}_stage_timing_cudagraph_report"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}" + (f"_{lmax_name}" if max_length is not None else ""))
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def make_cache_file_path(
    raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1
):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    raw_dir = Path(raw_data_dir)
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}_stage_timing_cg.json"
    )
    return raw_dir / fname


def save_raw_cache(path, meta: dict, lengths, threshold_ms, stage1_ms, stage2_ms, full_ms, skip_ratios):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "threshold_cg_ms": [float(x) for x in threshold_ms],
        "stage1_cg_ms": [float(x) for x in stage1_ms],
        "stage2_cg_ms": [float(x) for x in stage2_ms],
        "full_cg_ms": [float(x) for x in full_ms],
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
        data["threshold_cg_ms"],
        data["stage1_cg_ms"],
        data["stage2_cg_ms"],
        data["full_cg_ms"],
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


class CUDAGraphKernelRunner:
    def __init__(self, launch, warmup: int = 2):
        for _ in range(max(1, warmup)):
            launch()
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            launch()
        torch.cuda.synchronize()

    def replay(self):
        self._graph.replay()


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
    return info


def build_report(
    args,
    gpu_props,
    device_idx,
    gpu_tag,
    lengths,
    threshold_ms_list,
    stage1_ms_list,
    stage2_ms_list,
    full_ms_list,
    skip_ratios,
    meta,
):
    def _sum(a, b, c):
        if a is None or b is None or c is None:
            return None
        return float(a) + float(b) + float(c)

    stage_sum = [_sum(a, b, c) for a, b, c in zip(threshold_ms_list, stage1_ms_list, stage2_ms_list)]

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
        "threshold_cg_ms": threshold_ms_list,
        "stage1_cg_ms": stage1_ms_list,
        "stage2_cg_ms": stage2_ms_list,
        "full_cg_ms": full_ms_list,
        "stage_sum_ms": stage_sum,
        "skip_ratios": skip_ratios,
        "summary": {
            "length_last": lengths[-1] if lengths else None,
            "threshold_last": threshold_ms_list[-1] if threshold_ms_list else None,
            "stage1_last": stage1_ms_list[-1] if stage1_ms_list else None,
            "stage2_last": stage2_ms_list[-1] if stage2_ms_list else None,
            "stage_sum_last": stage_sum[-1] if stage_sum else None,
            "full_last": full_ms_list[-1] if full_ms_list else None,
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
    use_fp8_residual = not args.no_fp8_residual

    kernel_module = load_kernel_components(args.kernel)
    attn_forward_decode = kernel_module.attn_forward_decode_quantized
    attn_compute_threshold = kernel_module.attn_compute_threshold_qbits
    attn_stage1 = kernel_module.attn_forward_stage1_fused_threshold_qbits
    attn_stage2 = kernel_module.attn_forward_stage2_masked
    attn_kernel_name = kernel_module.__name__.split(".")[-1]

    try:
        from attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8_cudagraph import CUDAGraphDecodeRunnerQ2FP8
    except Exception as exc:
        raise RuntimeError(f"Failed to import CUDAGraphDecodeRunnerQ2FP8: {exc}") from exc

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
            raise ValueError(f"--length must be in [1, {T_full}], got {args.length}")
        lengths = [int(args.length)]
    else:
        lengths = list(range(step, T_full, step)) + [T_full]

    _, raw_data_dir = build_output_dirs(
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
    )

    if cache_path.exists() and not args.force:
        (
            x_lengths,
            threshold_ms_list,
            stage1_ms_list,
            stage2_ms_list,
            full_ms_list,
            skip_ratios,
            meta,
        ) = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        threshold_ms_list = []
        stage1_ms_list = []
        stage2_ms_list = []
        full_ms_list = []
        skip_ratios = []

        for L in tqdm(lengths, desc=f"delta={delta:g}, layers{layer_range_str}(bsz={bsz})"):
            q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
            k_rope = k_rope_full[:, :, :L, :].contiguous()
            v = v_full[:, :, :L, :].contiguous()

            q, k, v = convert_layout(q_rope_1, k_rope, v)
            q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
            k_q, k_scale, k_zero, k_residual = quantize_k_2bit_fp8_residual(k)

            B, _, HQ, D = q_1.shape
            T = k_q.shape[1]
            HKV = k_q.shape[2]
            K_packed = k_q.shape[3]
            G = HQ // HKV

            NTB = math.ceil(T / BS)
            NSB = math.ceil(BS / SBS)
            NTBS = NTB * NSB

            threshold_buf = torch.empty((B, HQ), device=q_1.device, dtype=torch.float32)
            m_buf = torch.empty((B, HQ, NTBS), device=q_1.device, dtype=torch.float32)
            l_buf = torch.empty((B, HQ, NTBS), device=q_1.device, dtype=torch.float32)
            o_buf = torch.empty((B, HQ, NTBS, V), device=q_1.device, dtype=torch.float32)
            mask_buf = torch.zeros((B, HKV, NTBS), device=q_1.device, dtype=torch.int8)
            o = torch.empty((B, HQ, V), device=q_1.device, dtype=q_1.dtype)

            if use_fp8_residual:
                k_res = k_residual
            else:
                k_res = k_q

            def launch_threshold():
                attn_compute_threshold[(B, HKV)](
                    q_1, k_q, k_scale, k_zero,
                    threshold_buf,
                    scale, T, NTB, delta,
                    B=B, HKV=HKV, HQ=HQ, K=D, K_PACKED=K_packed, G=G,
                    K_BITS=2,
                )

            def launch_stage1():
                attn_stage1[(NTB, B, HKV)](
                    q_1, k_q, k_scale, k_zero, k_res, v,
                    m_buf, l_buf, o_buf,
                    mask_buf,
                    scale, T, NTB, NTBS, delta,
                    threshold_buf,
                    B=B, HKV=HKV, HQ=HQ, K=D, K_PACKED=K_packed, V=V, G=G, BS=BS, SBS=SBS,
                    K_BITS=2, USE_EXT_TH=True, USE_FP8_RESIDUAL=use_fp8_residual,
                )

            def launch_stage2():
                attn_stage2[(B, HKV, G)](
                    m_buf, l_buf, o_buf,
                    mask_buf,
                    o, NTBS,
                    B=B, HKV=HKV, G=G, HQ=HQ, V=V,
                )

            threshold_runner = CUDAGraphKernelRunner(launch_threshold, warmup=args.cg_warmup)
            threshold_runner.replay()
            torch.cuda.synchronize()

            stage1_runner = CUDAGraphKernelRunner(launch_stage1, warmup=args.cg_warmup)
            stage1_runner.replay()
            torch.cuda.synchronize()

            stage2_runner = CUDAGraphKernelRunner(launch_stage2, warmup=args.cg_warmup)
            stage2_runner.replay()
            torch.cuda.synchronize()

            full_runner = CUDAGraphDecodeRunnerQ2FP8(
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
                use_fp8_residual=use_fp8_residual,
                warmup=args.cg_warmup,
            )

            def run_threshold():
                threshold_runner.replay()

            def run_stage1():
                stage1_runner.replay()

            def run_stage2():
                stage2_runner.replay()

            def run_full():
                return full_runner.replay_only()

            ms_threshold = benchmark(run_threshold, iters=iters, warmup=warmup)
            ms_stage1 = benchmark(run_stage1, iters=iters, warmup=warmup)

            stage1_runner.replay()
            torch.cuda.synchronize()
            ms_stage2 = benchmark(run_stage2, iters=iters, warmup=warmup)
            ms_full = benchmark(run_full, iters=iters, warmup=warmup)

            _, skip_ratio = attn_forward_decode(
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
                use_fp8_residual=use_fp8_residual,
            )

            threshold_ms_list.append(ms_threshold)
            stage1_ms_list.append(ms_stage1)
            stage2_ms_list.append(ms_stage2)
            full_ms_list.append(ms_full)
            skip_ratios.append(float(skip_ratio))

            print(
                f"[Stage-CG] T={to_k_str(L)} | th={ms_threshold:.3f} ms | "
                f"s1={ms_stage1:.3f} ms | s2={ms_stage2:.3f} ms | "
                f"sum={ms_threshold + ms_stage1 + ms_stage2:.3f} ms | "
                f"full={ms_full:.3f} ms | skip={skip_ratio:.2%}"
            )

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
            cg_warmup=int(args.cg_warmup),
            attn_kernel=attn_kernel_name,
            bsz=int(bsz),
            cudagraph=True,
            cudagraph_replay_only=True,
            use_fp8_residual=bool(use_fp8_residual),
        )
        save_raw_cache(
            cache_path,
            meta,
            x_lengths,
            threshold_ms_list,
            stage1_ms_list,
            stage2_ms_list,
            full_ms_list,
            skip_ratios,
        )
        print(f"[Info] Saved stage timing data to {cache_path}")

    report = build_report(
        args,
        gpu_props,
        gpu_idx,
        gpu_tag,
        x_lengths,
        threshold_ms_list,
        stage1_ms_list,
        stage2_ms_list,
        full_ms_list,
        skip_ratios,
        meta,
    )

    if args.out:
        report_path = Path(args.out)
    else:
        report_root = build_report_root(
            attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, THIS_DIR
        )
        Tmax_k_str = to_k_str(meta.get("T_full", 0))
        report_name = f"layer_layers_{layer_range_str}_Tmax{Tmax_k_str}_stage_timing_report.json"
        report_path = report_root / report_name

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[Result] Saved report to: {report_path}")


if __name__ == "__main__":
    main()

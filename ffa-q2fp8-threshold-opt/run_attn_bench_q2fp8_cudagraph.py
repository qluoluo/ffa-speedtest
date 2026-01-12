# Benchmarking & plotting for Q2FP8 decode with CUDAGraph.
import argparse
import importlib
import inspect
import json
import math
import re
import shutil
import sys
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

EXP_ROOT_DIR = THIS_DIR.parent / "attn_analysis" / "result"
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Q2FP8 decode with CUDAGraph.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128)
    p.add_argument("--SBS", type=int, default=None)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    p.add_argument(
        "--attn-kernel",
        type=str,
        default="attn_q2fp8_base_mask",
        help="Kernel module name under attn_kernel/ (e.g. attn_q2fp8_base_mask).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--step", type=int, default=4096, help="Step size for length sweep.")
    p.add_argument("--iters", type=int, default=500, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=100, help="Benchmark warmup")
    p.add_argument("--cg-warmup", type=int, default=2, help="CUDAGraph warmup calls before capture")
    p.add_argument(
        "--cg-replay-only",
        action="store_true",
        default=True,
        help="Measure CUDAGraph replay time only (exclude input copies).",
    )
    p.add_argument(
        "--with-q2",
        action="store_true",
        help="Also benchmark the non-CUDAGraph Q2FP8 baseline.",
    )
    p.add_argument("--num-warps", type=int, default=None, help="Override Triton num_warps for all kernels.")
    p.add_argument("--num-stages", type=int, default=None, help="Override Triton num_stages for all kernels.")
    p.add_argument("--num-warps-th", type=int, default=None, help="Override num_warps for threshold kernel.")
    p.add_argument("--num-stages-th", type=int, default=None, help="Override num_stages for threshold kernel.")
    p.add_argument("--num-warps-s1", type=int, default=None, help="Override num_warps for stage1 kernel.")
    p.add_argument("--num-stages-s1", type=int, default=None, help="Override num_stages for stage1 kernel.")
    p.add_argument("--num-warps-s2", type=int, default=None, help="Override num_warps for stage2 kernel.")
    p.add_argument("--num-stages-s2", type=int, default=None, help="Override num_stages for stage2 kernel.")
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    p.add_argument("--force", action="store_true", help="Force rerun and ignore cached results")
    p.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Profile internal Triton kernels at the maximum length.",
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
    # Scale/zero are per (B, HKV, K); token dimension is removed and broadcasted later.
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)
    k_dequant = (
        k_q.to(torch.float32) * scale[:, None, :, :].to(torch.float32) + zero[:, None, :, :].to(torch.float32)
    )
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack 4x2-bit values into a single byte to avoid storing each 2-bit value as uint8
    B, T, HKV, K = k_q.shape
    values_per_byte = 4  # 8 bits / 2 bits
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


def quantize_k_2bit_fp8_residual_symmetric(
    k: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e5m2,
    k_bits: int = 2,
):
    # Symmetric quantization per (B, HKV, K); token dimension is removed and broadcasted later.
    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    k_absmax = k.abs().amax(dim=1)
    scale = (k_absmax / qzero).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)
    k_dequant = (k_q.to(torch.float32) - qzero) * scale[:, None, :, :].to(torch.float32)
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack 4x2-bit values into a single byte to avoid storing each 2-bit value as uint8
    B, T, HKV, K = k_q.shape
    values_per_byte = 4  # 8 bits / 2 bits
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
    return k_q_packed, scale, k_residual


def _resolve_cudagraph_runner(cg_module):
    for name in (
        "CUDAGraphDecodeRunnerQ2FP8Vec",
        "CUDAGraphDecodeRunnerQ2FP8",
        "CUDAGraphDecodeRunnerQ2FP8ReuseBS",
    ):
        if hasattr(cg_module, name):
            return getattr(cg_module, name)
    for name, obj in vars(cg_module).items():
        if name.startswith("CUDAGraphDecodeRunner"):
            return obj
    raise AttributeError("No CUDAGraphDecodeRunner class found in cudagraph module.")


def load_kernel_components(kernel_path: str):
    kernel_name = kernel_path.strip()
    if kernel_name.endswith(".py"):
        kernel_name = kernel_name[:-3]
    if kernel_name.startswith("attn_kernel."):
        module_path = kernel_name
        kernel_name = kernel_name.split(".", 1)[1]
    else:
        module_path = f"attn_kernel.{kernel_name}"
    try:
        kernel_module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        kernel_dir = THIS_DIR / "attn_kernel"
        available = sorted(
            path.stem for path in kernel_dir.glob("*.py") if path.name != "__init__.py"
        )
        available_str = ", ".join(available) if available else "<empty>"
        raise ModuleNotFoundError(
            f"Kernel '{kernel_name}' not found under attn_kernel/. Available: {available_str}"
        ) from exc
    if not hasattr(kernel_module, "attn_forward_decode_quantized"):
        raise AttributeError(f"Module {module_path} does not define 'attn_forward_decode_quantized'")
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode_quantized")
    cudagraph_runner = _resolve_cudagraph_runner(kernel_module)
    return kernel_module, attn_forward_decode, cudagraph_runner


def resolve_quant_mode(kernel_module, attn_forward_decode_quantized, attn_kernel_name: str) -> str:
    mode = getattr(kernel_module, "QUANT_MODE", None)
    if mode is not None:
        norm = str(mode).strip().lower()
        if norm in ("sym", "symmetric"):
            return "sym"
        if norm in ("asym", "asymmetric"):
            return "asym"
        raise ValueError(f"Unsupported QUANT_MODE '{mode}' in {attn_kernel_name}")

    sig = inspect.signature(attn_forward_decode_quantized)
    if "k_zero" in sig.parameters:
        return "asym"
    if "sym" in attn_kernel_name:
        return "sym"
    return "sym"


def _filter_kwargs_for_signature(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


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


def build_plot_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, base_dir: Path):
    layer_range = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"
    lmax_name = str(max_length) if max_length is not None else ""
    plot_root_dir = (
        base_dir
        / "plot"
        / f"{attn_kernel_name}_cudagraph"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}" + (f"_{lmax_name}" if max_length is not None else ""))
    )
    raw_data_dir = plot_root_dir / "raw"
    return plot_root_dir, raw_data_dir


def make_cache_file_path(
    raw_data_dir,
    layer_idx,
    T_full,
    Hq,
    Hkv,
    D,
    Dv,
    BS,
    SBS,
    delta,
    dtype,
    step,
    iters,
    warmup,
    attn_kernel=None,
    bsz=1,
    replay_only=False,
    num_warps_th=None,
    num_stages_th=None,
    num_warps_s1=None,
    num_stages_s1=None,
    num_warps_s2=None,
    num_stages_s2=None,
):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"

    def _fmt(v):
        return "d" if v is None else str(v)
    raw_dir = Path(raw_data_dir)
    suffix = "_cudagraph_replay" if replay_only else "_cudagraph"
    kernel_tag = ""
    if any(
        v is not None
        for v in (
            num_warps_th,
            num_stages_th,
            num_warps_s1,
            num_stages_s1,
            num_warps_s2,
            num_stages_s2,
        )
    ):
        kernel_tag = (
            f"_nwT{_fmt(num_warps_th)}nsT{_fmt(num_stages_th)}"
            f"_nw1{_fmt(num_warps_s1)}ns1{_fmt(num_stages_s1)}"
            f"_nw2{_fmt(num_warps_s2)}ns2{_fmt(num_stages_s2)}"
        )
    kernel_name_tag = ""
    if attn_kernel:
        safe_name = str(attn_kernel).replace("/", "_")
        kernel_name_tag = f"_kernel{safe_name}"
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"{kernel_name_tag}{kernel_tag}_step{step}_it{iters}_wu{warmup}_bsz{bsz}{suffix}.json"
    )
    return raw_dir / fname


def save_raw_cache(path, meta: dict, lengths, q2_ms, q2_cg_ms, flash_ms, skip_ratios):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "q2_ms": [None if x is None else float(x) for x in q2_ms],
        "q2_cg_ms": [None if x is None else float(x) for x in q2_cg_ms],
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


def plot_curve(
    x_lengths,
    q2_ms_list,
    q2_cg_ms_list,
    flash_ms_list,
    T_full,
    BS,
    SBS,
    delta,
    layer_idx,
    out_dir,
    attn_kernel_name=None,
    skip_ratios=None,
    gpu_label=None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    line_q2_cg, = ax1.plot(
        x_lengths,
        q2_cg_ms_list,
        label="Q2FP8 CUDAGraph",
        marker="o",
        markersize=2,
        color="tab:purple",
    )
    lines = []
    labels = []
    if q2_ms_list is not None and any(x is not None for x in q2_ms_list):
        line_q2, = ax1.plot(
            x_lengths,
            q2_ms_list,
            label="Q2FP8",
            marker="o",
            markersize=2,
            color="tab:blue",
        )
        lines.append(line_q2)
        labels.append("Q2FP8")
    lines.append(line_q2_cg)
    labels.append("Q2FP8 CUDAGraph")

    if flash_ms_list is not None and any(x is not None for x in flash_ms_list):
        line_flash, = ax1.plot(
            x_lengths,
            flash_ms_list,
            label="FlashAttn",
            marker="o",
            markersize=2,
            color="tab:orange",
        )
        lines.append(line_flash)
        labels.append("FlashAttn")

    ax1.set_xlabel("Sequence length (T)")
    ax1.set_ylabel("Latency per run (ms)")
    Tmax_k_str = to_k_str(T_full)
    kernel_info = f" | Kernel: {attn_kernel_name}" if attn_kernel_name else ""
    ax1.set_title(
        f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta}{kernel_info})"
    )
    ax1.grid(True, linestyle="--", alpha=0.4)

    if gpu_label:
        ax1.text(
            0.01,
            0.99,
            f"GPU: {gpu_label}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if skip_ratios is not None:
        ax2 = ax1.twinx()
        skip_pct = [sr * 100.0 if sr is not None else 0.0 for sr in skip_ratios]
        line_skip, = ax2.plot(
            x_lengths,
            skip_pct,
            label="Skip ratio (%)",
            color="tab:green",
            linestyle="--",
            marker="x",
            markersize=2,
        )
        ax2.set_ylabel("Skip ratio (%)")
        ax2.set_ylim(0, 100)
        lines.append(line_skip)
        labels.append("Skip ratio (%)")

    ax1.legend(lines, labels)

    if attn_kernel_name:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_{attn_kernel_name}_cudagraph.png"
    else:
        plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_cudagraph.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    kernel_module, attn_forward_decode_quantized, cudagraph_runner = load_kernel_components(args.attn_kernel)
    attn_kernel_name = kernel_module.__name__.split(".")[-1]
    if attn_kernel_name.endswith("_cudagraph"):
        attn_kernel_name = attn_kernel_name[: -len("_cudagraph")]
    quant_mode = resolve_quant_mode(kernel_module, attn_forward_decode_quantized, attn_kernel_name)

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    step = int(args.step)
    iters = int(args.iters)
    warmup = int(args.warmup)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length
    run_q2 = bool(args.with_q2)

    def _norm_kernel_arg(val: int | None):
        if val is None:
            return None
        if val <= 0:
            return None
        return int(val)
    num_warps = _norm_kernel_arg(args.num_warps)
    num_stages = _norm_kernel_arg(args.num_stages)
    num_warps_th = _norm_kernel_arg(args.num_warps_th) or num_warps
    num_stages_th = _norm_kernel_arg(args.num_stages_th) or num_stages
    num_warps_s1 = _norm_kernel_arg(args.num_warps_s1) or num_warps
    num_stages_s1 = _norm_kernel_arg(args.num_stages_s1) or num_stages
    num_warps_s2 = _norm_kernel_arg(args.num_warps_s2) or num_warps
    num_stages_s2 = _norm_kernel_arg(args.num_stages_s2) or num_stages

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))
    layer_range_str = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")
    print(f"[Info] Quant mode: {quant_mode}")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    lengths = list(range(step, T_full, step)) + [T_full]

    flash_attn_compute, flash_err = maybe_load_flash(args.no_flash)
    if flash_attn_compute is None:
        print(f"[Info] FlashAttention baseline disabled: {flash_err}")

    def build_quant_inputs(length: int):
        q_rope_1 = q_rope_full[:, :, length - 1 : length, :].contiguous()
        k_rope = k_rope_full[:, :, :length, :].contiguous()
        v = v_full[:, :, :length, :].contiguous()

        q, k, v = convert_layout(q_rope_1, k_rope, v)
        q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
        k_q_asym, k_scale_asym, k_zero_asym, k_residual_asym = quantize_k_2bit_fp8_residual(k)
        k_q_sym, k_scale_sym, k_residual_sym = quantize_k_2bit_fp8_residual_symmetric(k)

        if quant_mode == "sym":
            k_q = k_q_sym
            k_scale = k_scale_sym
            k_zero = None
            k_residual = k_residual_sym
        else:
            k_q = k_q_asym
            k_scale = k_scale_asym
            k_zero = k_zero_asym
            k_residual = k_residual_asym

        quant_inputs = dict(
            q=q_1,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_residual=k_residual,
        )
        if k_zero is not None:
            quant_inputs["k_zero"] = k_zero
        return q, k, v, quant_inputs

    def profile_internal_kernels(length: int, quant_inputs: dict, *, warmup: bool = False):
        profile_kwargs = dict(
            **quant_inputs,
            k_bits=2,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_kernel_timings=True,
            num_warps_th=num_warps_th,
            num_stages_th=num_stages_th,
            num_warps_s1=num_warps_s1,
            num_stages_s1=num_stages_s1,
            num_warps_s2=num_warps_s2,
            num_stages_s2=num_stages_s2,
        )
        if warmup:
            warmup_kwargs = dict(profile_kwargs)
            warmup_kwargs.pop("return_kernel_timings", None)
            attn_forward_decode_quantized(
                **_filter_kwargs_for_signature(attn_forward_decode_quantized, warmup_kwargs)
            )
        profile_out = attn_forward_decode_quantized(
            **_filter_kwargs_for_signature(attn_forward_decode_quantized, profile_kwargs)
        )
        kernel_profile = None
        if isinstance(profile_out, tuple):
            if len(profile_out) == 2:
                _, kernel_profile = profile_out
            elif len(profile_out) == 3:
                _, _, kernel_profile = profile_out
        if kernel_profile is None:
            return length, None
        total_ms = sum(ms for ms in kernel_profile.values() if ms is not None)
        pct = {}
        if total_ms > 0:
            for name, ms in kernel_profile.items():
                pct[name] = None if ms is None else (ms / total_ms) * 100.0
        else:
            for name in kernel_profile.keys():
                pct[name] = None
        profile = {
            "length": int(length),
            "total_ms": float(total_ms),
            "ms": kernel_profile,
            "pct": pct,
        }
        return length, profile

    plot_root_dir, raw_data_dir = build_plot_dirs(
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
        attn_kernel=attn_kernel_name,
        bsz=bsz,
        replay_only=args.cg_replay_only,
        num_warps_th=num_warps_th,
        num_stages_th=num_stages_th,
        num_warps_s1=num_warps_s1,
        num_stages_s1=num_stages_s1,
        num_warps_s2=num_warps_s2,
        num_stages_s2=num_stages_s2,
    )

    created_plot_dir = False
    created_raw_dir = False
    kernel_profile = None
    kernel_profile_length = None

    def ensure_plot_dir():
        nonlocal created_plot_dir
        if not plot_root_dir.exists():
            plot_root_dir.mkdir(parents=True, exist_ok=True)
            created_plot_dir = True

    def ensure_raw_dir():
        nonlocal created_raw_dir
        ensure_plot_dir()
        if not raw_data_dir.exists():
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            created_raw_dir = True

    try:
        use_cache = cache_path.exists() and not args.force
        cached_q2_ms_list = None
        if use_cache:
            (
                x_lengths,
                cached_q2_ms_list,
                q2_cg_ms_list,
                flash_ms_list,
                skip_ratios,
                _meta,
            ) = load_raw_cache(cache_path)
            cache_has_q2 = bool(cached_q2_ms_list) and all(x is not None for x in cached_q2_ms_list)
            if run_q2 and not cache_has_q2:
                print(f"[Info] Cached results missing Q2 baseline; rerunning to collect it.")
                use_cache = False
        if use_cache:
            if run_q2:
                q2_ms_list = cached_q2_ms_list
            else:
                q2_ms_list = [None] * len(q2_cg_ms_list)
            print(f"[Info] Loaded cached results from {cache_path}")
            if args.profile_kernels:
                _, _, _, quant_inputs = build_quant_inputs(lengths[-1])
                kernel_profile_length, kernel_profile = profile_internal_kernels(
                    lengths[-1], quant_inputs, warmup=True
                )
                if kernel_profile is not None and "kernel_profile" not in _meta:
                    _meta["kernel_profile"] = kernel_profile
                    save_raw_cache(cache_path, _meta, x_lengths, q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios)
                    print(f"[Info] Updated cached results with kernel profile at {cache_path}")
        else:
            if cache_path.exists() and args.force:
                print(f"[Info] Force rerun enabled; ignoring cached results at {cache_path}")
            q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios = [], [], [], []

            for L in tqdm(lengths, desc=f"delta={delta:g}, layers{layer_range_str}(bsz={bsz})"):
                q, k, v, quant_inputs = build_quant_inputs(L)

                def _run_attn(return_skip_ratio: bool):
                    call_kwargs = dict(
                        **quant_inputs,
                        k_bits=2,
                        scale=scale,
                        BS=BS,
                        SBS=SBS,
                        delta=delta,
                        return_skip_ratio=return_skip_ratio,
                        num_warps_th=num_warps_th,
                        num_stages_th=num_stages_th,
                        num_warps_s1=num_warps_s1,
                        num_stages_s1=num_stages_s1,
                        num_warps_s2=num_warps_s2,
                        num_stages_s2=num_stages_s2,
                    )
                    return attn_forward_decode_quantized(
                        **_filter_kwargs_for_signature(attn_forward_decode_quantized, call_kwargs)
                    )

                # One forward to obtain skip ratio and validate shapes
                _, skip_ratio = _run_attn(return_skip_ratio=True)

                runner_kwargs = dict(
                    **quant_inputs,
                    k_bits=2,
                    scale=scale,
                    BS=BS,
                    SBS=SBS,
                    delta=delta,
                    use_fp8_residual=True,
                    warmup=args.cg_warmup,
                    num_warps_th=num_warps_th,
                    num_stages_th=num_stages_th,
                    num_warps_s1=num_warps_s1,
                    num_stages_s1=num_stages_s1,
                    num_warps_s2=num_warps_s2,
                    num_stages_s2=num_stages_s2,
                )
                runner = cudagraph_runner(
                    **_filter_kwargs_for_signature(cudagraph_runner.__init__, runner_kwargs),
                )

                def run_q2():
                    return _run_attn(return_skip_ratio=False)

                def run_q2_cg():
                    if args.cg_replay_only:
                        return runner.replay_only()
                    return runner(
                        **_filter_kwargs_for_signature(runner.replay, quant_inputs),
                    )

                def run_flash():
                    return flash_attn_compute(q, k, v)

                ms_q2 = None
                if run_q2:
                    ms_q2 = benchmark(run_q2, iters=iters, warmup=warmup)
                ms_q2_cg = benchmark(run_q2_cg, iters=iters, warmup=warmup)
                ms_flash = None
                if flash_attn_compute is not None:
                    ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

                q2_ms_list.append(ms_q2)
                q2_cg_ms_list.append(ms_q2_cg)
                flash_ms_list.append(ms_flash)
                skip_ratios.append(float(skip_ratio))

                if args.profile_kernels and L == lengths[-1]:
                    kernel_profile_length, kernel_profile = profile_internal_kernels(L, quant_inputs)

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
                quant_mode=quant_mode,
                bsz=int(bsz),
                cudagraph=True,
                cudagraph_replay_only=bool(args.cg_replay_only),
                q2_baseline=bool(run_q2),
                num_warps_th=num_warps_th,
                num_stages_th=num_stages_th,
                num_warps_s1=num_warps_s1,
                num_stages_s1=num_stages_s1,
                num_warps_s2=num_warps_s2,
                num_stages_s2=num_stages_s2,
            )
            if kernel_profile is not None:
                meta["kernel_profile"] = kernel_profile
            ensure_raw_dir()
            save_raw_cache(cache_path, meta, x_lengths, q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios)
            print(f"[Info] Saved raw benchmark data to {cache_path}")

        plot_path = None
        if not args.no_plot:
            ensure_plot_dir()
            plot_path = plot_curve(
                x_lengths,
                q2_ms_list,
                q2_cg_ms_list,
                flash_ms_list,
                T_full,
                BS,
                SBS,
                delta,
                f"layers_{layer_range_str}_bsz_{bsz}",
                plot_root_dir,
                attn_kernel_name,
                skip_ratios=skip_ratios,
                gpu_label=gpu_label,
            )

        speedup_parts = []
        if flash_ms_list[-1] is not None:
            if q2_ms_list[-1] is not None and q2_ms_list[-1] > 0:
                speedup = flash_ms_list[-1] / q2_ms_list[-1]
                speedup_parts.append(f"Speedup={speedup:.2f}x")
            if q2_cg_ms_list[-1] > 0:
                speedup_cg = flash_ms_list[-1] / q2_cg_ms_list[-1]
                speedup_parts.append(f"Speedup_CG={speedup_cg:.2f}x")
        speedup_str = f", {', '.join(speedup_parts)}" if speedup_parts else ""
        print(
            f"[Result] Layers {layer_range_str} | bsz={bsz} | T={to_k_str(T_full)} | "
            f"BS={BS} SBS={SBS} delta={delta} | "
            + (f"Q2={q2_ms_list[-1]:.3f} ms, " if q2_ms_list[-1] is not None else "")
            + f"Q2_CG={q2_cg_ms_list[-1]:.3f} ms"
            + (f", Flash={flash_ms_list[-1]:.3f} ms" if flash_ms_list[-1] is not None else "")
            + speedup_str
        )
        if kernel_profile is not None:
            parts = []
            total_ms = float(kernel_profile.get("total_ms", 0.0))
            ms_map = kernel_profile.get("ms", {})
            pct_map = kernel_profile.get("pct", {})
            for name, ms in ms_map.items():
                if ms is None:
                    parts.append(f"{name}=skipped")
                    continue
                pct = pct_map.get(name)
                if pct is None or total_ms <= 0:
                    parts.append(f"{name}={ms:.3f} ms")
                else:
                    parts.append(f"{name}={ms:.3f} ms ({pct:.1f}%)")
            if total_ms > 0:
                parts.append(f"total={total_ms:.3f} ms")
            if parts:
                prof_len = to_k_str(kernel_profile_length) if kernel_profile_length else "n/a"
                print(f"[KernelProfile] T={prof_len} | " + ", ".join(parts))
        elif args.profile_kernels:
            print("[KernelProfile] No kernel timings reported for this kernel module.")
        if plot_path is not None:
            print(f"[Result] Saved plot to: {plot_path}")
    except Exception:
        if created_plot_dir and plot_root_dir.exists():
            shutil.rmtree(plot_root_dir, ignore_errors=True)
        elif created_raw_dir and raw_data_dir.exists():
            shutil.rmtree(raw_data_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

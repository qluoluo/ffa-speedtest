# Benchmarking & plotting for FP8 + FP8 residual attention kernel.
import argparse
import importlib
import math
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import dtype_key, load_raw_cache, make_cache_file_path, save_raw_cache, to_k_str
from utils.flash import flash_attn_compute
from utils.load import load_qkvh
from utils.plot import plot_speed_curve

# Ensure attn_kernel package is importable
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Default kernel for fp8 + fp8 residual
from attn_kernel.attn_kernel_v1210_fused_bsz_fp8fp8 import attn_forward_decode_fp8fp8

EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Run quantized attn kernel test with recorded layer data.")
    p.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1210_fused_bsz_fp8fp8",
        help="Python module path for attn_forward_decode_fp8fp8",
    )
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128)
    p.add_argument("--SBS", type=int, default=None)
    p.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help="Delta value for skipping; run the script once per delta to compare (e.g., 3, 5, 8, 10).",
    )
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--step", type=int, default=1024, help="Step size for length sweep when plotting.")
    p.add_argument("--iters", type=int, default=100, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup")
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


def _resolve_fp8_dtype(device: torch.device) -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        try:
            torch.empty(1, device=device, dtype=torch.float8_e5m2)
            return torch.float8_e5m2
        except Exception:
            pass
    return torch.float16


def quantize_k_fp8_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype | None = None):
    if fp8_dtype is None:
        fp8_dtype = _resolve_fp8_dtype(k.device)
    k_base = k.to(fp8_dtype).contiguous()
    k_residual = (k.to(torch.float32) - k_base.to(torch.float32)).to(fp8_dtype).contiguous()
    return k_base, k_residual, fp8_dtype


def load_kernel_components(kernel_path: str):
    kernel_module = importlib.import_module(kernel_path)
    if not hasattr(kernel_module, "attn_forward_decode_fp8fp8"):
        raise AttributeError(f"Module {kernel_path} does not define 'attn_forward_decode_fp8fp8'")
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode_fp8fp8")
    return kernel_module, attn_forward_decode


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
        / f"{attn_kernel_name}"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}" + (f"_{lmax_name}" if max_length is not None else ""))
    )
    plot_root_dir.mkdir(parents=True, exist_ok=True)

    raw_data_dir = plot_root_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return plot_root_dir, raw_data_dir


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

    kernel_module, attn_forward_decode = load_kernel_components(args.kernel)
    attn_kernel_name = kernel_module.__name__.split(".")[-1]

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))
    layer_range_str = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)
    fp8_dtype = _resolve_fp8_dtype(q_rope_full.device)
    if not hasattr(torch, "float8_e5m2") or fp8_dtype != torch.float8_e5m2:
        print("[Info] fp8 dtype not available on this device, using fp16 base/residuals.")

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    lengths = list(range(step, T_full, step)) + [T_full]

    def bench_one_length(L, delta):
        q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
        k_rope = k_rope_full[:, :, :L, :].contiguous()
        v = v_full[:, :, :L, :].contiguous()

        q, k, v = convert_layout(q_rope_1, k_rope, v)
        q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
        k_fp8, k_residual, _fp8_dtype = quantize_k_fp8_fp8_residual(k, fp8_dtype=fp8_dtype)

        def run_fp8fp8():
            return attn_forward_decode(
                q=q_1,
                k_fp8=k_fp8,
                k_residual=k_residual,
                v=v,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                return_skip_ratio=False,
            )

        def run_flash():
            return flash_attn_compute(q, k, v)

        # One forward to obtain skip ratio and validate shapes
        _, skip_ratio = attn_forward_decode(
            q=q_1,
            k_fp8=k_fp8,
            k_residual=k_residual,
            v=v,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
        )

        ms_fp8fp8 = benchmark(run_fp8fp8, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fp8fp8, ms_flash, float(skip_ratio)

    def validate_full(delta):
        q_rope_1 = q_rope_full[:, :, T_full - 1 : T_full, :].contiguous()
        q, k, v = convert_layout(q_rope_1, k_rope_full, v_full)
        q_1 = q.unsqueeze(1)
        k_fp8, k_residual, _fp8_dtype = quantize_k_fp8_fp8_residual(k, fp8_dtype=fp8_dtype)

        o_triton, skip_ratio = attn_forward_decode(
            q=q_1,
            k_fp8=k_fp8,
            k_residual=k_residual,
            v=v,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
        )
        o_flash = flash_attn_compute(q, k, v)
        max_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().mean().item()
        print(
            f"[Validate] delta={delta} | skip_ratio={skip_ratio:.3%} | "
            f"max_abs={max_abs_vs_flash:.3e}, mean_abs={mean_abs_vs_flash:.3e}"
        )

    print(f"[Info] Running delta={delta} | layers={layer_range_str} | bsz={bsz}")
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
        bsz=bsz,
    )

    if cache_path.exists():
        x_lengths, fp8_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        fp8_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc=f"delta={delta:g}, layers{layer_range_str}(bsz={bsz})"):
            ms_fp8fp8, ms_flash, sr = bench_one_length(L, delta)
            fp8_ms_list.append(ms_fp8fp8)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)
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
        )
        save_raw_cache(cache_path, meta, x_lengths, fp8_ms_list, flash_ms_list, skip_ratios)
        print(f"[Info] Saved raw benchmark data to {cache_path}")

    plot_path = plot_speed_curve(
        x_lengths,
        fp8_ms_list,
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
    print(
        f"[Result] Layers {layer_range_str} | bsz={bsz} | T={to_k_str(T_full)} | BS={BS} SBS={SBS} delta={delta} | "
        f"FP8FP8={fp8_ms_list[-1]:.3f} ms, Flash={flash_ms_list[-1]:.3f} ms"
    )
    print(f"[Result] Saved plot to: {plot_path}")
    validate_full(delta)


if __name__ == "__main__":
    main()

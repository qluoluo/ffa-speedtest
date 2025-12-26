# Benchmarking & plotting for NF4 K + fp8 residual attention kernel.
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

# Default kernel for nf4 + fp8 residual
from attn_kernel.attn_kernel_v1210_fused_bsz_nf4fp8 import attn_forward_decode_nf4

EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


NF4_CODEBOOK_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5229921340942383,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

_NF4_TABLE_CACHE = {}


def _get_nf4_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index)
    if key in _NF4_TABLE_CACHE:
        return _NF4_TABLE_CACHE[key]
    codebook = torch.tensor(NF4_CODEBOOK_VALUES, device=device, dtype=torch.float32)
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    _NF4_TABLE_CACHE[key] = (codebook, boundaries)
    return codebook, boundaries


def parse_args():
    p = argparse.ArgumentParser(description="Run NF4 attn kernel test with recorded layer data.")
    p.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1210_fused_bsz_nf4fp8",
        help="Python module path for attn_forward_decode_nf4",
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


def encode_k_nf4_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codebook, boundaries = _get_nf4_tables(k.device)
    k_f = k.to(torch.float32)
    scale = k_f.abs().amax(dim=1).clamp_min(1e-6)
    scale_out = scale.to(k.dtype).contiguous()
    norm = (k_f / scale[:, None, :, :]).clamp(codebook[0].item(), codebook[-1].item())
    idx = torch.bucketize(norm, boundaries).to(torch.int64)
    k_nf4 = idx.to(torch.uint8).contiguous()
    k_dequant = codebook[idx] * scale[:, None, :, :]
    k_residual = (k_f - k_dequant).to(fp8_dtype).contiguous()

    B, T, HKV, K = k_nf4.shape
    values_per_byte = 2
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_nf4.device, dtype=k_nf4.dtype)
        k_nf4 = torch.cat([k_nf4, pad_tensor], dim=-1)
    k_nf4 = k_nf4.view(B, T, HKV, k_packed_len, values_per_byte)
    k_nf4_packed = (k_nf4[..., 0] | (k_nf4[..., 1] << 4)).contiguous()
    return k_nf4_packed, scale_out, k_residual


def load_kernel_components(kernel_path: str):
    kernel_module = importlib.import_module(kernel_path)
    if not hasattr(kernel_module, "attn_forward_decode_nf4"):
        raise AttributeError(f"Module {kernel_path} does not define 'attn_forward_decode_nf4'")
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode_nf4")
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
        print("[Info] fp8 dtype not available on this device, using fp16 residuals.")

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    lengths = list(range(step, T_full, step)) + [T_full]

    def bench_one_length(L, delta):
        q_rope_1 = q_rope_full[:, :, :1, :]
        k_rope = k_rope_full[:, :, :L, :]
        v = v_full[:, :, :L, :]
        q, k, v = convert_layout(q_rope_1, k_rope, v)

        k_nf4, k_scale, k_residual = encode_k_nf4_fp8_residual(k, fp8_dtype)

        def run_nf4():
            return attn_forward_decode(
                q=q,
                k_nf4=k_nf4,
                k_scale=k_scale,
                k_residual=k_residual,
                v=v,
                k_bits=4,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                return_skip_ratio=False,
                use_fp8_residual=True,
            )

        _, skip_ratio = attn_forward_decode(
            q=q,
            k_nf4=k_nf4,
            k_scale=k_scale,
            k_residual=k_residual,
            v=v,
            k_bits=4,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
            use_fp8_residual=True,
        )

        ms_nf4 = benchmark(run_nf4, iters=iters, warmup=warmup)

        ms_flash = flash_attn_compute(q, k, v, iters=iters, warmup=warmup)
        return ms_nf4, ms_flash, float(skip_ratio)

    def prewarm_kv():
        q_rope_1 = q_rope_full[:, :, :1, :]
        k_rope = k_rope_full[:, :, : min(step, T_full), :]
        v = v_full[:, :, : min(step, T_full), :]
        q, k, v = convert_layout(q_rope_1, k_rope, v)
        k_nf4, k_scale, k_residual = encode_k_nf4_fp8_residual(k, fp8_dtype)
        attn_forward_decode(
            q=q,
            k_nf4=k_nf4,
            k_scale=k_scale,
            k_residual=k_residual,
            v=v,
            k_bits=4,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=False,
            use_fp8_residual=True,
        )

    prewarm_kv()

    plot_root_dir, raw_data_dir = build_plot_dirs(
        attn_kernel_name,
        gpu_tag,
        BS,
        SBS,
        delta,
        layer_indices,
        bsz,
        max_length,
        base_dir=THIS_DIR,
    )
    cache_path = make_cache_file_path(
        raw_data_dir, layer_range_str, T_full, Hq, Hkv, K, V, BS, SBS, delta, dtype, step, iters, warmup, bsz
    )

    if cache_path.exists():
        x_lengths, nf4_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached data from {cache_path.name}")
    else:
        nf4_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc="Benchmarking"):
            ms_nf4, ms_flash, sr = bench_one_length(L, delta)
            nf4_ms_list.append(ms_nf4)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)

        meta = {
            "layer": layer_range_str,
            "dtype": dtype_key(dtype),
            "BS": BS,
            "SBS": SBS,
            "delta": delta,
            "step": step,
            "iters": iters,
            "warmup": warmup,
            "gpu": gpu_label,
        }
        save_raw_cache(cache_path, meta, lengths, nf4_ms_list, flash_ms_list, skip_ratios)
        print(f"[Info] Saved raw data to {cache_path.name}")

    plot_path = plot_speed_curve(
        x_lengths,
        nf4_ms_list,
        flash_ms_list,
        T_full,
        BS,
        SBS,
        delta,
        layer_range_str,
        plot_root_dir,
        attn_kernel_name=attn_kernel_name,
        skip_ratios=skip_ratios,
    )

    print(f"[Result] Plot saved at: {plot_path}")
    print(
        f"[Info] Done: layer={layer_range_str} last T={to_k_str(T_full)} | "
        f"NF4={nf4_ms_list[-1]:.3f} ms, Flash={flash_ms_list[-1]:.3f} ms"
    )


if __name__ == "__main__":
    main()

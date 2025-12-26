# Benchmarking & plotting for FP4 K + fp8 residual decode with CUDAGraph.
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import dtype_key, to_k_str
from utils.load import load_qkvh

from attn_kernel.attn_kernel_v1210_fused_bsz_fp4fp8 import attn_forward_decode_quantized
from attn_kernel.attn_kernel_v1210_fused_bsz_fp4fp8_cudagraph import (
    CUDAGraphDecodeRunnerFP4FP8,
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
    p = argparse.ArgumentParser(description="Benchmark FP4FP8 decode with CUDAGraph.")
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
    p.add_argument("--step", type=int, default=1024, help="Step size for length sweep.")
    p.add_argument("--iters", type=int, default=500, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=100, help="Benchmark warmup")
    p.add_argument("--cg-warmup", type=int, default=2, help="CUDAGraph warmup calls before capture")
    p.add_argument(
        "--cg-replay-only",
        action="store_true",
        help="Measure CUDAGraph replay time only (exclude input copies).",
    )
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
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


def quantize_k_4bit_fp8_residual(k: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2):
    # FP4 E2M1 (bias=1) positive levels: [0, 0.5, 1, 1.5, 2, 3, 4, 6].
    fp4_pos = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=k.device,
        dtype=k.dtype,
    )
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=k.device,
        dtype=k.dtype,
    )

    abs_k = k.abs()
    idx = torch.bucketize(abs_k, thresholds, right=True).to(torch.int64)
    sign = (k < 0)
    k_q = (idx.to(torch.uint8) | (sign.to(torch.uint8) << 3)).contiguous()
    sign_scale = torch.where(sign, -torch.ones_like(abs_k), torch.ones_like(abs_k))
    k_dequant = fp4_pos[idx] * sign_scale
    k_residual = (k - k_dequant).to(fp8_dtype).contiguous()

    # Dummy scale/zero for interface compatibility (unused by FP4 kernel).
    B, T, HKV, K = k.shape
    k_scale = torch.ones((B, HKV, K), device=k.device, dtype=k.dtype)
    k_zero = torch.zeros((B, HKV, K), device=k.device, dtype=k.dtype)

    # Pack 2x4-bit values into a single byte to avoid storing each 4-bit value as uint8
    values_per_byte = 2  # 8 bits / 4 bits
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)
        k_q = torch.cat([k_q, pad_tensor], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (k_q[..., 0] | (k_q[..., 1] << 4)).contiguous()
    return k_q_packed, k_scale, k_zero, k_residual


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
    plot_root_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir = plot_root_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return plot_root_dir, raw_data_dir


def make_cache_file_path(raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1, replay_only=False):
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


def save_raw_cache(path, meta: dict, lengths, fp4_ms, fp4_cg_ms, flash_ms, skip_ratios):
    path = Path(path)
    payload = {
        "meta": meta,
        "lengths": [int(x) for x in lengths],
        "fp4_ms": [float(x) for x in fp4_ms],
        "fp4_cg_ms": [float(x) for x in fp4_cg_ms],
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
        data["fp4_ms"],
        data["fp4_cg_ms"],
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
    fp4_ms_list,
    fp4_cg_ms_list,
    flash_ms_list,
    T_full,
    BS,
    SBS,
    delta,
    layer_idx,
    out_dir,
    attn_kernel_name=None,
    skip_ratios=None,
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

    line_fp4, = ax1.plot(x_lengths, fp4_ms_list, label="FP4FP8", marker="o", markersize=2)
    line_fp4_cg, = ax1.plot(x_lengths, fp4_cg_ms_list, label="FP4FP8 CUDAGraph", marker="o", markersize=2)
    lines = [line_fp4, line_fp4_cg]
    labels = ["FP4FP8", "FP4FP8 CUDAGraph"]

    if flash_ms_list is not None and any(x is not None for x in flash_ms_list):
        line_flash, = ax1.plot(
            x_lengths,
            flash_ms_list,
            label="FlashAttn",
            marker="o",
            markersize=2,
            color="tab:purple",
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

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    step = int(args.step)
    iters = int(args.iters)
    warmup = int(args.warmup)
    bsz = int(args.bsz)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    attn_kernel_name = "attn_kernel_v1210_fused_bsz_fp4fp8"

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))
    layer_range_str = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    lengths = list(range(step, T_full, step)) + [T_full]

    flash_attn_compute, flash_err = maybe_load_flash(args.no_flash)
    if flash_attn_compute is None:
        print(f"[Info] FlashAttention baseline disabled: {flash_err}")

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
        replay_only=args.cg_replay_only,
    )

    if cache_path.exists():
        x_lengths, fp4_ms_list, fp4_cg_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        fp4_ms_list, fp4_cg_ms_list, flash_ms_list, skip_ratios = [], [], [], []

        for L in tqdm(lengths, desc=f"delta={delta:g}, layers{layer_range_str}(bsz={bsz})"):
            q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
            k_rope = k_rope_full[:, :, :L, :].contiguous()
            v = v_full[:, :, :L, :].contiguous()

            q, k, v = convert_layout(q_rope_1, k_rope, v)
            q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
            k_q, k_scale, k_zero, k_residual = quantize_k_4bit_fp8_residual(k)

            # One forward to obtain skip ratio and validate shapes
            _, skip_ratio = attn_forward_decode_quantized(
                q=q_1,
                k_q=k_q,
                k_scale=k_scale,
                k_zero=k_zero,
                k_residual=k_residual,
                v=v,
                k_bits=4,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                return_skip_ratio=True,
            )

            runner = CUDAGraphDecodeRunnerFP4FP8(
                q_1,
                k_q,
                k_scale,
                k_zero,
                v,
                k_residual=k_residual,
                k_bits=4,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                use_fp8_residual=True,
                warmup=args.cg_warmup,
            )

            def run_fp4():
                return attn_forward_decode_quantized(
                    q=q_1,
                    k_q=k_q,
                    k_scale=k_scale,
                    k_zero=k_zero,
                    k_residual=k_residual,
                    v=v,
                    k_bits=4,
                    scale=scale,
                    BS=BS,
                    SBS=SBS,
                    delta=delta,
                    return_skip_ratio=False,
                )

            def run_fp4_cg():
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

            ms_fp4 = benchmark(run_fp4, iters=iters, warmup=warmup)
            ms_fp4_cg = benchmark(run_fp4_cg, iters=iters, warmup=warmup)
            ms_flash = None
            if flash_attn_compute is not None:
                ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

            fp4_ms_list.append(ms_fp4)
            fp4_cg_ms_list.append(ms_fp4_cg)
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
        save_raw_cache(cache_path, meta, x_lengths, fp4_ms_list, fp4_cg_ms_list, flash_ms_list, skip_ratios)
        print(f"[Info] Saved raw benchmark data to {cache_path}")

    plot_path = None
    if not args.no_plot:
        plot_path = plot_curve(
            x_lengths,
            fp4_ms_list,
            fp4_cg_ms_list,
            flash_ms_list,
            T_full,
            BS,
            SBS,
            delta,
            f"layers_{layer_range_str}_bsz_{bsz}",
            plot_root_dir,
            attn_kernel_name,
            skip_ratios=skip_ratios,
        )

    print(
        f"[Result] Layers {layer_range_str} | bsz={bsz} | T={to_k_str(T_full)} | "
        f"BS={BS} SBS={SBS} delta={delta} | "
        f"FP4={fp4_ms_list[-1]:.3f} ms, FP4_CG={fp4_cg_ms_list[-1]:.3f} ms"
        + (f", Flash={flash_ms_list[-1]:.3f} ms" if flash_ms_list[-1] is not None else "")
    )
    if plot_path is not None:
        print(f"[Result] Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()

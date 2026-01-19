#!/usr/bin/env python3
# Benchmarking & plotting for Q2FP8 Unified Kernel with CUDAGraph
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directory to path for imports
THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from utils.bench import benchmark
from utils.cache import dtype_key, to_k_str, make_cache_file_path, save_raw_cache, load_raw_cache

# Import unified kernel
sys.path.insert(0, str(PARENT_DIR / "e2e" / "q2fp8-unified"))
from attn_kernel.attn_q2fp8_unified import attn_forward_decode_quantized, CUDAGraphDecodeRunnerQ2FP8

# Data path
EXP_ROOT_DIR = Path(
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Q2FP8 Unified Kernel with CUDAGraph.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128, help="Block size for quantized blocks")
    p.add_argument("--SBS", type=int, default=None, help="Sub-block size")
    p.add_argument("--delta", type=float, default=5.0, help="Threshold delta")
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    p.add_argument("--current-len", type=int, default=64, help="Number of FP16 current tokens (0-128)")
    p.add_argument("--max-current", type=int, default=128, help="Max current buffer size")
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
        default=True,
        help="Measure CUDAGraph replay time only (exclude input copies).",
    )
    p.add_argument(
        "--with-q2",
        action="store_true",
        help="Also benchmark the non-CUDAGraph Q2FP8 baseline.",
    )
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    p.add_argument("--force", action="store_true", help="Force rerun and ignore cached results")
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def convert_layout(q_rope_1: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor):
    """Convert from [B, H, T, D] to [B, T, H, D] layout."""
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv
    q = q_rope_1[:, :, 0, :].contiguous()  # [B, Hq, D]
    k = k_rope.permute(0, 2, 1, 3).contiguous()  # [B, T, Hkv, D]
    v = v.permute(0, 2, 1, 3).contiguous()  # [B, T, Hkv, D]
    return q, k, v


def quantize_k_symmetric_2bit(k: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2):
    """Symmetric 2-bit quantization with FP8 residual."""
    # Scale per (B, HKV, K); token dimension removed
    k_min = k.amin(dim=1, keepdim=True)  # [B, 1, HKV, K]
    k_max = k.amax(dim=1, keepdim=True)  # [B, 1, HKV, K]
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).squeeze(1).contiguous()  # [B, HKV, K]

    # Symmetric quantization: no zero point
    k_center = (k_max + k_min) / 2.0
    k_q = torch.round((k - k_center) / scale[:, None, :, :] + 1.5).clamp(0, 3).to(torch.uint8)

    # Compute residual
    k_dequant = (k_q.to(torch.float32) - 1.5) * scale[:, None, :, :].to(torch.float32) + k_center.to(torch.float32)
    k_residual = (k.to(torch.float32) - k_dequant).to(fp8_dtype).contiguous()

    # Pack 4x2-bit values into a single byte
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


def build_plot_dirs(gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, current_len, base_dir: Path):
    layer_range = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"
    lmax_name = str(max_length) if max_length is not None else ""
    plot_root_dir = (
        base_dir
        / "plot"
        / "q2fp8_unified_cudagraph"
        / gpu_tag
        / (f"delta{delta}_layers{layer_range}_BS{BS}_SBS{SBS}_bsz{bsz}_curr{current_len}" + (f"_{lmax_name}" if lmax_name else ""))
    )
    raw_data_dir = plot_root_dir / "raw"
    return plot_root_dir, raw_data_dir


def load_qkvh(layer_data_root, device="cuda", start_layer=0, max_length=None):
    """Generator to load layer data."""
    layer_data_root = Path(layer_data_root)
    layer_idx = start_layer
    while True:
        layer_dir = layer_data_root / f"layer_{layer_idx}"
        if not layer_dir.exists():
            break

        q_rope_path = layer_dir / "q_rope.pt"
        k_rope_path = layer_dir / "k_rope.pt"
        v_path = layer_dir / "v.pt"

        if not (q_rope_path.exists() and k_rope_path.exists() and v_path.exists()):
            break

        q_rope = torch.load(q_rope_path, map_location=device)
        k_rope = torch.load(k_rope_path, map_location=device)
        v = torch.load(v_path, map_location=device)

        if max_length is not None and max_length > 0:
            k_rope = k_rope[:, :, :max_length, :]
            v = v[:, :, :max_length, :]

        yield {"q_rope": q_rope, "k_rope": k_rope, "v": v}
        layer_idx += 1


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
    skip_ratios=None,
    gpu_label=None,
    current_len=0,
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
        label=f"Q2FP8 Unified CUDAGraph (curr={current_len})",
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
            label=f"Q2FP8 Unified (curr={current_len})",
            marker="o",
            markersize=2,
            color="tab:blue",
        )
        lines.append(line_q2)
        labels.append(f"Q2FP8 Unified (curr={current_len})")
    lines.append(line_q2_cg)
    labels.append(f"Q2FP8 Unified CUDAGraph (curr={current_len})")

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
    ax1.set_title(
        f"Q2FP8 Unified CUDAGraph | Layer {layer_idx} | Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta}, curr={current_len}"
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

    plot_path = out_dir / f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_unified_cudagraph_curr{current_len}.png"

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
    current_len = int(args.current_len)
    max_current = int(args.max_current)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length
    run_q2 = bool(args.with_q2)

    if not (0 <= current_len <= max_current):
        raise ValueError(f"current_len must be in [0, {max_current}], got {current_len}")

    exp_root = EXP_ROOT_DIR / EXP_ROOT_SUBDIR
    layer_data_root = exp_root / "layer_data"
    layer_indices = list(range(args.layer, args.layer + bsz))
    layer_range_str = f"{layer_indices[0]}" if len(layer_indices) == 1 else f"{layer_indices[0]}-{layer_indices[-1]}"

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    gpu_label = f"{gpu_name} ({gpu_mem_gb}GB)"
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_label}")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, max_length)

    bsz_actual, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layers={layer_indices}, bsz={bsz_actual}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")
    print(f"[Info] Current tokens: {current_len} (max buffer: {max_current})")

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

        # Split k/v into quantized and current parts
        if current_len > 0 and length > current_len:
            k_quantized = k[:, :-current_len, :, :].contiguous()
            v_quantized = v[:, :-current_len, :, :].contiguous()
            k_curr = k[:, -current_len:, :, :].contiguous()
            v_curr = v[:, -current_len:, :, :].contiguous()

            # Pad to max_current
            B, curr_len_actual, HKV, K_dim = k_curr.shape
            if curr_len_actual < max_current:
                k_current = torch.zeros((B, max_current, HKV, K_dim), device=k_curr.device, dtype=k_curr.dtype)
                v_current = torch.zeros((B, max_current, HKV, V), device=v_curr.device, dtype=v_curr.dtype)
                k_current[:, :curr_len_actual, :, :] = k_curr
                v_current[:, :curr_len_actual, :, :] = v_curr
            else:
                k_current = k_curr
                v_current = v_curr

            actual_current_len = curr_len_actual
        else:
            k_quantized = k
            v_quantized = v
            k_current = torch.zeros((bsz_actual, max_current, Hkv, K), device=k.device, dtype=k.dtype)
            v_current = torch.zeros((bsz_actual, max_current, Hkv, V), device=v.device, dtype=v.dtype)
            actual_current_len = 0

        k_q, k_scale, k_residual = quantize_k_symmetric_2bit(k_quantized)

        quant_inputs = dict(
            q=q_1,
            k_q=k_q,
            k_scale=k_scale,
            v=v_quantized,
            k_current=k_current,
            v_current=v_current,
            current_len=actual_current_len,
            k_residual=k_residual,
        )
        return q, k, v, quant_inputs

    plot_root_dir, raw_data_dir = build_plot_dirs(
        gpu_tag, BS, SBS, delta, layer_indices, bsz, max_length, current_len, THIS_DIR
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
        current_len=current_len,
        cudagraph=True,
        replay_only=args.cg_replay_only,
    )

    created_plot_dir = False
    created_raw_dir = False

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
        else:
            if cache_path.exists() and args.force:
                print(f"[Info] Force rerun enabled; ignoring cached results at {cache_path}")
            q2_ms_list, q2_cg_ms_list, flash_ms_list, skip_ratios = [], [], [], []

            for L in tqdm(lengths, desc=f"delta={delta:g}, layers{layer_range_str}(bsz={bsz}), curr={current_len}"):
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
                        use_fp8_residual=True,
                        max_current=max_current,
                    )
                    return attn_forward_decode_quantized(**call_kwargs)

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
                    max_current=max_current,
                )
                runner = CUDAGraphDecodeRunnerQ2FP8(**runner_kwargs)

                def run_q2():
                    return _run_attn(return_skip_ratio=False)

                def run_q2_cg():
                    if args.cg_replay_only and hasattr(runner, "replay_only"):
                        return runner.replay_only()
                    return runner.replay(**quant_inputs)

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
                attn_kernel="q2fp8_unified",
                bsz=int(bsz),
                current_len=int(current_len),
                max_current=int(max_current),
                cudagraph=True,
                cudagraph_replay_only=bool(args.cg_replay_only),
                q2_baseline=bool(run_q2),
            )
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
                skip_ratios=skip_ratios,
                gpu_label=gpu_label,
                current_len=current_len,
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
            f"[Result] Layers {layer_range_str} | bsz={bsz} | T={to_k_str(T_full)} | curr={current_len} | "
            f"BS={BS} SBS={SBS} delta={delta} | "
            + (f"Q2={q2_ms_list[-1]:.3f} ms, " if q2_ms_list[-1] is not None else "")
            + f"Q2_CG={q2_cg_ms_list[-1]:.3f} ms"
            + (f", Flash={flash_ms_list[-1]:.3f} ms" if flash_ms_list[-1] is not None else "")
            + speedup_str
        )
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

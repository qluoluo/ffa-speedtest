# Benchmark paged attention (decode) with q2 packed K.
import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import make_cache_file_path, save_attn_cache, load_attn_cache, to_k_str
from utils.load import load_qkvh
from utils.plot import plot_speed_curve

from attn_kernel.attn_q2fp8_paged import (
    allocate_paged_kv_cache,
    update_pages,
    update_block_table,
)
from attn_kernel.attn_q2fp8_paged_attn import PagedAttnRunner

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

EXP_ROOT_DIR = THIS_DIR.parent / "attn_analysis" / "result"
EXP_ROOT_SUBDIR = Path("Llama-3_2-3B/longbench_gov_report_48_68_256k")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark paged attention decode.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--SBS", type=int, default=256, help="Page size (tokens per page)")
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--bsz", type=int, default=1, help="Batch size (layers to combine)")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--step", type=int, default=4096, help="Step size for length sweep.")
    p.add_argument("--iters", type=int, default=200, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup")
    p.add_argument("--delta", type=float, default=5.0, help="Threshold delta (larger => less prune)")
    p.add_argument("--num-warps", type=int, default=4, help="Triton num_warps for stage1")
    p.add_argument("--num-stages", type=int, default=2, help="Triton num_stages for stage1")
    p.add_argument("--num-warps-th", type=int, default=None, help="Triton num_warps for threshold kernel")
    p.add_argument("--num-stages-th", type=int, default=None, help="Triton num_stages for threshold kernel")
    p.add_argument("--num-warps-s2", type=int, default=None, help="Triton num_warps for stage2")
    p.add_argument("--num-stages-s2", type=int, default=None, help="Triton num_stages for stage2")
    p.add_argument("--no-flash", action="store_true", help="Skip FlashAttention baseline")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    p.add_argument("--force", action="store_true", help="Force rerun and ignore cached results")
    p.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help="Override experiment root (default: attn_analysis/result)",
    )
    p.add_argument(
        "--exp-subdir",
        type=str,
        default=None,
        help="Override experiment subdir (default: Llama-3_2-3B/...)",
    )
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
        # Assume [B, Hkv, T, D]
        k = k_rope.permute(0, 2, 1, 3).contiguous()
        v_out = v.permute(0, 2, 1, 3).contiguous()
    elif dim2 == dim2v:
        # Assume [B, T, Hkv, D]
        k = k_rope.contiguous()
        v_out = v.contiguous()
    else:
        raise ValueError("k_rope/v layouts must be [B, Hkv, T, D] or [B, T, Hkv, D]")

    return q, k, v_out


def quantize_k_2bit_symmetric_packed(k: torch.Tensor, k_bits: int = 2):
    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    k_absmax = k.abs().amax(dim=1)
    scale = (k_absmax / qzero).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)

    values_per_byte = 8 // k_bits
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
    return k_q_packed, scale


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


def build_pages(k_q_packed: torch.Tensor, v: torch.Tensor, page_size: int):
    B, T, HKV, K_PACKED = k_q_packed.shape
    _, _, _, V = v.shape
    n_pages = math.ceil(T / page_size)
    pad = n_pages * page_size - T
    if pad:
        pad_k = torch.zeros((B, pad, HKV, K_PACKED), device=k_q_packed.device, dtype=k_q_packed.dtype)
        pad_v = torch.zeros((B, pad, HKV, V), device=v.device, dtype=v.dtype)
        k_q_packed = torch.cat([k_q_packed, pad_k], dim=1)
        v = torch.cat([v, pad_v], dim=1)

    k_pages = k_q_packed.view(B, n_pages, page_size, HKV, K_PACKED)
    k_pages = k_pages.permute(0, 1, 3, 2, 4).contiguous()

    v_pages = v.view(B, n_pages, page_size, HKV, V)
    v_pages = v_pages.permute(0, 1, 3, 2, 4).contiguous()

    page_lens = torch.full((B, n_pages), page_size, device=v.device, dtype=torch.int32)
    if pad:
        page_lens[:, -1] = page_size - pad

    return k_pages, v_pages, page_lens, n_pages


def sanitize_gpu_name(name: str) -> str:
    return name.replace(" ", "-").replace("/", "_")


def maybe_load_flash(no_flash: bool):
    if no_flash:
        return None, "disabled"
    try:
        from utils.flash import flash_attn_compute
        import flash_attn  # noqa: F401
        return flash_attn_compute, None
    except Exception as exc:
        return None, str(exc)


def add_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def main():
    args = parse_args()

    if args.bsz != 1:
        raise ValueError("Current paged attention benchmark only supports bsz=1.")

    dtype = map_dtype(args.dtype)
    exp_root = Path(args.exp_root) if args.exp_root else EXP_ROOT_DIR
    exp_subdir = Path(args.exp_subdir) if args.exp_subdir else EXP_ROOT_SUBDIR
    layer_data_root = exp_root / exp_subdir / "layer_data"

    layer_indices = list(range(args.layer, args.layer + args.bsz))
    q_rope_full, k_rope_full, v_full = load_layer_batch(
        layer_data_root, layer_indices, dtype, args.max_length
    )

    q_rope_1 = q_rope_full[:, :, :1, :].contiguous()
    _, k, v = convert_layout(q_rope_1, k_rope_full, v_full)

    B, HQ, T_full, K = q_rope_full.shape
    _, _, HKV, _ = k.shape
    V = v.shape[-1]

    if args.max_length is not None:
        if args.max_length > 0:
            T_full = min(T_full, args.max_length)
        elif args.max_length < 0:
            T_full = T_full

    lengths = [T_full]
    if args.step > 0:
        lengths = list(range(args.step, T_full + 1, args.step))
        if lengths[-1] != T_full:
            lengths.append(T_full)

    gpu_name = sanitize_gpu_name(torch.cuda.get_device_name(0))
    run_tag = f"layers{args.bsz}_SBS{args.SBS}_delta{args.delta}_bsz{args.bsz}"
    plot_root = THIS_DIR / "plot" / "paged_attn" / gpu_name / run_tag
    raw_data_dir = plot_root / "raw"

    cache_path = make_cache_file_path(
        raw_data_dir,
        args.layer,
        T_full,
        HQ,
        HKV,
        K,
        V,
        args.SBS,
        dtype,
        args.step,
        args.iters,
        args.warmup,
        bsz=args.bsz,
    )
    cache_path = add_suffix(cache_path, f"_delta{args.delta}")

    flash_attn_fn, flash_err = maybe_load_flash(args.no_flash)
    if flash_attn_fn is None and not args.no_flash:
        print(f"[Warn] FlashAttn not available: {flash_err}")

    if cache_path.exists() and not args.force:
        lengths, paged_ms_list, flash_ms_list, meta = load_attn_cache(cache_path)
    else:
        paged_ms_list = []
        flash_ms_list = []
        meta = {
            "dtype": args.dtype,
            "SBS": args.SBS,
            "delta": args.delta,
            "iters": args.iters,
            "warmup": args.warmup,
            "layer": args.layer,
            "bsz": args.bsz,
            "flash_baseline": flash_attn_fn is not None,
            "num_warps_s1": args.num_warps,
            "num_stages_s1": args.num_stages,
            "num_warps_th": args.num_warps_th,
            "num_stages_th": args.num_stages_th,
            "num_warps_s2": args.num_warps_s2,
            "num_stages_s2": args.num_stages_s2,
        }

        for L in tqdm(lengths, desc=f"paged_attn layers{args.bsz}"):
            q = q_rope_full[:, :, L - 1, :].contiguous()
            k_slice = k[:, :L].contiguous()
            v_slice = v[:, :L].contiguous()

            k_q_packed, k_scale = quantize_k_2bit_symmetric_packed(k_slice)
            k_pages, v_pages, page_lens, n_pages = build_pages(k_q_packed, v_slice, args.SBS)

            cache = allocate_paged_kv_cache(
                max_pages=n_pages,
                page_size=args.SBS,
                num_kv_heads=HKV,
                head_dim=K,
                value_dim=V,
                k_scale=k_scale,
                v_dtype=dtype,
                device=k_slice.device,
                max_batch=args.bsz,
                max_pages_per_seq=n_pages,
            )

            page_ids = torch.arange(n_pages, device=k_slice.device, dtype=torch.int32)
            update_block_table(cache, seq_ids=[0] * n_pages, page_ids=page_ids.tolist())

            # Update paged cache (metadata disabled for timing).
            update_pages(
                cache,
                page_ids,
                k_pages[0],
                v_pages[0],
                page_lens=page_lens[0],
                compute_meta=False,
            )

            runner = PagedAttnRunner(
                q,
                cache,
                delta=args.delta,
                num_warps_s1=args.num_warps,
                num_stages_s1=args.num_stages,
                num_warps_th=args.num_warps_th,
                num_stages_th=args.num_stages_th,
                num_warps_s2=args.num_warps_s2,
                num_stages_s2=args.num_stages_s2,
            )

            # Trigger compilation outside timing.
            _ = runner.run()
            torch.cuda.synchronize()

            paged_ms = benchmark(runner.run, iters=args.iters, warmup=args.warmup)
            if flash_attn_fn is not None:
                flash_ms = benchmark(lambda: flash_attn_fn(q, k_slice, v_slice), iters=args.iters, warmup=args.warmup)
            else:
                flash_ms = None

            paged_ms_list.append(paged_ms)
            flash_ms_list.append(flash_ms)

        save_attn_cache(cache_path, meta, lengths, paged_ms_list, flash_ms_list)

    plot_path = None
    if not args.no_plot:
        plot_path = plot_speed_curve(
            lengths,
            paged_ms_list,
            flash_ms_list,
            T_full,
            args.SBS,
            args.delta,
            args.layer,
            plot_root,
            kernel_name="paged_attn_q2",
            gpu_label=gpu_name,
        )

    if paged_ms_list:
        msg = (
            f"[Result] Layer {args.layer} | bsz={args.bsz} | T={to_k_str(T_full)} | "
            f"SBS={args.SBS} | delta={args.delta} | paged={paged_ms_list[-1]:.3f} ms"
        )
        if flash_ms_list and flash_ms_list[-1] is not None:
            msg += f", flash={flash_ms_list[-1]:.3f} ms"
        print(msg)
    if plot_path is not None:
        print(f"[Result] Saved plot to: {plot_path}")
    print(f"[Info] Saved raw benchmark data to {cache_path}")


if __name__ == "__main__":
    main()

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
KERNEL_ROOT = THIS_DIR.parent
if str(KERNEL_ROOT) not in sys.path:
    sys.path.append(str(KERNEL_ROOT))

from transformers import AutoTokenizer

from q2fp8_cache import Q2Fp8Cache, Q2Fp8StaticCache, resolve_fp8_dtype
from modeling_llama import LlamaForCausalLM

INPUT_TEXT_PATH = THIS_DIR / "input_text.txt"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end benchmark for q2+fp8 cache on a Llama model.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument(
        "--model-path",
        type=str,
        default="/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B",
        help="Local path to a Llama model checkpoint.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--mode", type=str, default="both", choices=["prefill", "decode", "both"])
    p.add_argument("--compare", action="store_true", help="Compare baseline flash attention vs q2+fp8 decode.")
    p.add_argument("--ffa-decode", action="store_true", help="Enable q2+fp8 FFA decode kernel path.")
    p.add_argument(
        "--decode-tokens",
        type=int,
        default=1,
        help="Number of decode steps after prefill (q_len=1). Defaults to 1.",
    )
    p.add_argument("--greedy-decode", action="store_true", help="Greedy argmax decode each step.")
    p.add_argument("--bs", type=int, default=128, help="Block size for q2+fp8 decode kernel.")
    p.add_argument("--sbs", type=int, default=None, help="Sub-block size for q2+fp8 decode kernel.")
    p.add_argument("--delta", type=float, default=5.0, help="Delta threshold for q2+fp8 decode kernel.")
    p.add_argument("--cudagraph", action="store_true", help="Use CUDA Graphs for prefill (cache disabled).")
    p.add_argument("--decode-cudagraph", action="store_true", help="Use CUDA Graphs for decode (static cache).")
    p.add_argument(
        "--decode-kernel-cudagraph",
        action="store_true",
        help="Use kernel-level CUDA Graphs for q2+fp8 decode (static cache).",
    )
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]
    if device.type == "cpu" and dtype != torch.float32:
        print("[Warn] CPU device forces fp32; overriding dtype.")
        dtype = torch.float32
    return dtype


def load_input_text() -> str:
    if not INPUT_TEXT_PATH.exists():
        raise FileNotFoundError(f"Missing input text file: {INPUT_TEXT_PATH}")
    return INPUT_TEXT_PATH.read_text(encoding="utf-8")


def encode_input_text(text: str, tokenizer: AutoTokenizer, seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        return torch.empty((0,), dtype=torch.long)
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=seq_len,
    )
    if not token_ids:
        raise ValueError("Tokenizer produced no tokens from input text.")
    if seq_len > len(token_ids):
        raise ValueError(
            f"Input text too short for seq_len={seq_len} (tokens={len(token_ids)})."
        )
    return torch.tensor(token_ids, dtype=torch.long)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def benchmark_ms(fn, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync_device(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        sync_device(device)
        return start.elapsed_time(end) / iters

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def benchmark_cudagraph(fn, iters: int, warmup: int, device: torch.device) -> float:
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph benchmarking requires a CUDA device.")
    for _ in range(warmup):
        fn()
    sync_device(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    sync_device(device)
    return start.elapsed_time(end) / iters


def benchmark_decode_cudagraph(make_fn, iters: int, warmup: int, device: torch.device) -> float:
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph benchmarking requires a CUDA device.")
    # Ensure CUDA libraries (e.g., cuBLAS) are initialized outside graph capture.
    _ = torch.zeros((1, 1), device=device) @ torch.zeros((1, 1), device=device)
    sync_device(device)
    for _ in range(warmup):
        run_fn, _ = make_fn()
        run_fn()
    sync_device(device)

    run_fn, _ = make_fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    sync_device(device)
    return start.elapsed_time(end) / iters


def benchmark_decode_steps(
    build_prefill,
    run_decode,
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        cache, next_token = build_prefill()
        run_decode(cache, next_token)
    sync_device(device)

    if device.type == "cuda":
        total_ms = 0.0
        for _ in range(iters):
            cache, next_token = build_prefill()
            sync_device(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_decode(cache, next_token)
            end.record()
            sync_device(device)
            total_ms += start.elapsed_time(end)
        return total_ms / iters

    total_ms = 0.0
    for _ in range(iters):
        cache, next_token = build_prefill()
        start = time.perf_counter()
        run_decode(cache, next_token)
        end = time.perf_counter()
        total_ms += (end - start) * 1000.0
    return total_ms / iters


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.decode_tokens <= 0:
        raise ValueError("decode-tokens must be positive.")
    decode_steps = args.decode_tokens
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    use_residual = not args.no_residual
    if (args.ffa_decode or args.compare) and device.type != "cuda":
        raise RuntimeError("q2+fp8 FFA decode requires a CUDA device.")
    if args.mode in ("decode", "both") and decode_steps <= 0:
        raise ValueError("decode-tokens must be positive for decode mode.")

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device)
    model.eval()
    config = model.config
    if hasattr(config, "max_position_embeddings"):
        max_needed = args.seq_len + decode_steps
        if max_needed > config.max_position_embeddings:
            print(
                f"[Warn] seq_len+decode_tokens={max_needed} exceeds max_position_embeddings="
                f"{config.max_position_embeddings}."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if args.greedy_decode:
        required_len = max(args.seq_len, 1)
    else:
        required_len = args.seq_len + decode_steps
    input_tokens = encode_input_text(load_input_text(), tokenizer, required_len)
    input_ids = input_tokens.unsqueeze(0).repeat(args.batch, 1).to(device)
    fp8_dtype = resolve_fp8_dtype(device)
    prefill_ids = input_ids[:, : args.seq_len] if args.seq_len > 0 else None
    decode_ids = None
    if not args.greedy_decode and decode_steps > 0:
        decode_ids = input_ids[:, args.seq_len : args.seq_len + decode_steps]

    print(
        "[Info] "
        f"device={device.type} dtype={dtype} fp8_dtype={fp8_dtype} "
        f"batch={args.batch} seq_len={args.seq_len} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} kv_heads={config.num_key_value_heads} "
        f"layers={config.num_hidden_layers} "
        f"use_residual={use_residual} compare={args.compare} ffa_decode={args.ffa_decode} "
        f"decode_tokens={decode_steps} greedy={args.greedy_decode} "
        f"cudagraph={args.cudagraph} decode_cudagraph={args.decode_cudagraph} "
        f"decode_kernel_cudagraph={args.decode_kernel_cudagraph} "
        f"model_path={args.model_path}"
    )

    if args.cudagraph and args.mode in ("decode", "both") and not args.decode_cudagraph:
        print("[Warn] cudagraph only applies to prefill; use --decode-cudagraph for decode.")
    if args.decode_cudagraph or args.decode_kernel_cudagraph:
        raise ValueError(
            "Q2FP8 e2e does not support --decode-cudagraph/--decode-kernel-cudagraph."
        )

    pattern_layers = list(range(1, config.num_hidden_layers)) if config.num_hidden_layers > 1 else []
    head_dim = config.hidden_size // config.num_attention_heads

    def set_attn_settings(use_ffa_decode: bool) -> None:
        if use_ffa_decode:
            config.attn_settings = {
                "use_ffa_decode": True,
                "BS": args.bs,
                "SBS": args.sbs,
                "delta": args.delta,
                "use_fp8_residual": use_residual,
                "k_bits": 2,
                "pattern_layers": pattern_layers,
            }
        else:
            config.attn_settings = {}

    def run_prefill(use_cache: bool) -> None:
        with torch.no_grad():
            if args.seq_len <= 0:
                return
            if use_cache:
                cache = Q2Fp8Cache(
                    use_fp8_residual=use_residual,
                    fp8_dtype=fp8_dtype,
                )
                model(
                    prefill_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                model(
                    prefill_ids,
                    past_key_values=None,
                    use_cache=False,
                )

    def init_decode_runtime(cache) -> None:
        return

    def build_prefill_dynamic():
        cache = Q2Fp8Cache(
            use_fp8_residual=use_residual,
            fp8_dtype=fp8_dtype,
        )
        token_buf = None
        if args.greedy_decode:
            token_buf = torch.empty((args.batch, 1), device=device, dtype=torch.long)
        with torch.no_grad():
            if args.seq_len > 0:
                outputs = model(
                    prefill_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                if args.greedy_decode:
                    torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True, out=token_buf)
            elif args.greedy_decode:
                token_buf.copy_(input_ids[:, :1])
        init_decode_runtime(cache)
        return cache, token_buf

    def run_decode_steps(cache, token_buf) -> None:
        if decode_steps <= 0:
            return
        with torch.no_grad():
            if args.greedy_decode:
                for _ in range(decode_steps):
                    outputs = model(
                        token_buf,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True, out=token_buf)
            else:
                for t in range(decode_steps):
                    model(
                        decode_ids[:, t : t + 1],
                        past_key_values=cache,
                        use_cache=True,
                    )

    def make_decode_static():
        cache = Q2Fp8StaticCache(
            max_seq_len=args.seq_len + decode_steps,
            use_fp8_residual=use_residual,
            fp8_dtype=fp8_dtype,
        )
        token_buf = None
        if args.greedy_decode:
            token_buf = torch.empty((args.batch, 1), device=device, dtype=torch.long)
        with torch.no_grad():
            if args.seq_len > 0:
                outputs = model(
                    prefill_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                if args.greedy_decode:
                    torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True, out=token_buf)
            elif args.greedy_decode:
                token_buf.copy_(input_ids[:, :1])
        init_decode_runtime(cache)

        def run_decode_static() -> None:
            run_decode_steps(cache, token_buf)

        return run_decode_static, cache

    def report_skip_ratio(tag: str) -> None:
        if args.mode not in ("decode", "both") or not config.attn_settings.get("use_ffa_decode", False):
            return
        skip_ratios: list[float] = []
        config.attn_settings["return_skip_ratio"] = True
        config.attn_settings["skip_ratio_store"] = skip_ratios
        cache, token_buf = build_prefill_dynamic()
        run_decode_steps(cache, token_buf)
        config.attn_settings.pop("return_skip_ratio", None)
        config.attn_settings.pop("skip_ratio_store", None)
        if skip_ratios:
            avg_skip = sum(skip_ratios) / len(skip_ratios)
            print(f"[{tag}][Decode][SkipRatio] avg={avg_skip:.4f} samples={len(skip_ratios)}")

    def benchmark_prefill(tag: str) -> float | None:
        if args.mode not in ("prefill", "both"):
            return None
        if args.cudagraph:
            prefill_ms = benchmark_cudagraph(
                lambda: run_prefill(use_cache=False), args.iters, args.warmup, device
            )
            cache_note = "no-cache"
        else:
            prefill_ms = benchmark_ms(lambda: run_prefill(use_cache=True), args.iters, args.warmup, device)
            cache_note = "cache"
        tokens = args.batch * args.seq_len
        prefill_tps = tokens / (prefill_ms / 1000.0)
        print(f"[{tag}][Prefill][{cache_note}] {prefill_ms:.3f} ms/iter, {prefill_tps:.1f} tok/s")
        return prefill_ms

    def benchmark_decode(tag: str) -> float | None:
        if args.mode not in ("decode", "both"):
            return None
        if args.decode_cudagraph:
            decode_ms = benchmark_decode_cudagraph(make_decode_static, args.iters, args.warmup, device)
        else:
            decode_ms = benchmark_decode_steps(
                build_prefill_dynamic,
                run_decode_steps,
                args.iters,
                args.warmup,
                device,
            )
        tokens = args.batch * decode_steps
        decode_tps = tokens / (decode_ms / 1000.0)
        per_token = decode_ms / decode_steps
        print(f"[{tag}][Decode] {decode_ms:.3f} ms/iter, {per_token:.3f} ms/token, {decode_tps:.1f} tok/s")
        return decode_ms

    def run_variant(tag: str, use_ffa_decode: bool) -> None:
        set_attn_settings(use_ffa_decode)
        clear_cuda()
        prefill_ms = benchmark_prefill(tag)
        clear_cuda()
        report_skip_ratio(tag)
        clear_cuda()
        decode_ms = benchmark_decode(tag)
        if prefill_ms is not None and decode_ms is not None:
            total_ms = prefill_ms + decode_ms
            print(f"[{tag}][Total] {total_ms:.3f} ms/iter (prefill+decode)")
        clear_cuda()

    if args.compare:
        run_variant("Baseline", use_ffa_decode=False)
        run_variant("Q2FP8", use_ffa_decode=True)
    elif args.ffa_decode:
        run_variant("Q2FP8", use_ffa_decode=True)
    else:
        run_variant("Baseline", use_ffa_decode=False)


if __name__ == "__main__":
    main()

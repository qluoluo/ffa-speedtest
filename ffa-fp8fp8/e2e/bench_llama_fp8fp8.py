import argparse
import gc
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from fp8fp8_cache import FP8Fp8Cache, resolve_fp8_dtype
from modeling_llama import LlamaConfig, LlamaForCausalLM

INPUT_TEXT_PATH = THIS_DIR / "input_text.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end benchmark for fp8+fp8 cache on a Llama model.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--intermediate-size", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--mode", type=str, default="both", choices=["prefill", "decode", "both"])
    p.add_argument("--compare", action="store_true", help="Compare baseline flash attention vs fp8+fp8 decode.")
    p.add_argument("--ffa-decode", action="store_true", help="Enable fp8+fp8 FFA decode kernel path.")
    p.add_argument("--bs", type=int, default=128, help="Block size for fp8+fp8 decode kernel.")
    p.add_argument("--sbs", type=int, default=None, help="Sub-block size for fp8+fp8 decode kernel.")
    p.add_argument("--delta", type=float, default=5.0, help="Delta threshold for fp8+fp8 decode kernel.")
    p.add_argument("--cudagraph", action="store_true", help="Use CUDA Graphs for prefill (cache disabled).")
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


def encode_input_text(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError("vocab-size must be positive.")
    token_ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    if vocab_size < 256:
        token_ids = token_ids % vocab_size
    if seq_len > 0:
        if token_ids.numel() < seq_len:
            reps = (seq_len + token_ids.numel() - 1) // token_ids.numel()
            token_ids = token_ids.repeat(reps)
        token_ids = token_ids[:seq_len]
    return token_ids


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    use_residual = not args.no_residual
    if (args.ffa_decode or args.compare) and device.type != "cuda":
        raise RuntimeError("fp8+fp8 FFA decode requires a CUDA device.")

    if args.hidden_size % args.heads != 0:
        raise ValueError("hidden-size must be divisible by heads.")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv-heads.")

    intermediate_size = args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * 4
    max_position = max(args.seq_len, 64)

    config = LlamaConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        num_hidden_layers=args.layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position,
        vocab_size=args.vocab_size,
        attention_bias=False,
        mlp_bias=False,
    )

    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    input_tokens = encode_input_text(load_input_text(), config.vocab_size, args.seq_len)
    input_ids = input_tokens.unsqueeze(0).repeat(args.batch, 1).to(device)
    fp8_dtype = resolve_fp8_dtype(device)

    print(
        "[Info] "
        f"device={device.type} dtype={dtype} fp8_dtype={fp8_dtype} "
        f"batch={args.batch} seq_len={args.seq_len} hidden={args.hidden_size} "
        f"heads={args.heads} kv_heads={args.kv_heads} layers={args.layers} "
        f"use_residual={use_residual} compare={args.compare} ffa_decode={args.ffa_decode} "
        f"cudagraph={args.cudagraph}"
    )

    if args.cudagraph and args.mode in ("decode", "both"):
        print("[Warn] cudagraph only applies to prefill; decode runs in regular mode.")

    def set_attn_settings(use_ffa_decode: bool) -> None:
        if use_ffa_decode:
            config.attn_settings = {
                "use_ffa_decode": True,
                "BS": args.bs,
                "SBS": args.sbs,
                "delta": args.delta,
                "use_fp8_residual": use_residual,
            }
        else:
            config.attn_settings = {}

    def run_prefill(use_cache: bool) -> None:
        with torch.no_grad():
            if use_cache:
                cache = FP8Fp8Cache(
                    use_fp8_residual=use_residual,
                    fp8_dtype=fp8_dtype,
                )
                model(
                    input_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                model(
                    input_ids,
                    past_key_values=None,
                    use_cache=False,
                )

    def run_decode() -> None:
        with torch.no_grad():
            cache = FP8Fp8Cache(
                use_fp8_residual=use_residual,
                fp8_dtype=fp8_dtype,
            )
            for t in range(args.seq_len):
                model(
                    input_ids[:, t : t + 1],
                    past_key_values=cache,
                    use_cache=True,
                )

    def benchmark_prefill(tag: str) -> None:
        if args.mode not in ("prefill", "both"):
            return
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

    def benchmark_decode(tag: str) -> None:
        if args.mode not in ("decode", "both"):
            return
        decode_ms = benchmark_ms(run_decode, args.iters, args.warmup, device)
        tokens = args.batch * args.seq_len
        decode_tps = tokens / (decode_ms / 1000.0)
        per_token = decode_ms / args.seq_len
        print(f"[{tag}][Decode] {decode_ms:.3f} ms/iter, {per_token:.3f} ms/token, {decode_tps:.1f} tok/s")

    def run_variant(tag: str, use_ffa_decode: bool) -> None:
        set_attn_settings(use_ffa_decode)
        clear_cuda()
        benchmark_prefill(tag)
        clear_cuda()
        benchmark_decode(tag)
        clear_cuda()

    if args.compare:
        run_variant("Baseline", use_ffa_decode=False)
        run_variant("FP8FP8", use_ffa_decode=True)
    elif args.ffa_decode:
        run_variant("FP8FP8", use_ffa_decode=True)
    else:
        run_variant("Baseline", use_ffa_decode=False)


if __name__ == "__main__":
    main()

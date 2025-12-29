#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer, LlamaForCausalLM as HfLlamaForCausalLM

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from modeling_llama import LlamaForCausalLM as LocalLlamaForCausalLM

DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B"
DEFAULT_INPUT_PATH = THIS_DIR / "generate_input.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy generation with local modeling_llama and transformers Llama.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local path to a Llama model checkpoint.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the prompt text file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print the full prompt + continuation instead of only new tokens.",
    )
    return parser.parse_args()


def resolve_device_and_dtype(device_str: str, dtype_str: str) -> Tuple[torch.device, torch.dtype]:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    device = torch.device(device_str)
    dtype = dtype_map[dtype_str]
    if device.type == "cpu" and dtype != torch.float32:
        print("[Warn] CPU device forces fp32; overriding dtype.")
        dtype = torch.float32
    return device, dtype


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing input file: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input file is empty: {prompt_path}")
    return text


def prepare_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_cls, model_path: str, dtype: torch.dtype, device: torch.device):
    model = model_cls.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def greedy_generate(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> Tuple[str, str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    input_len = input_ids.shape[1]
    full_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    new_text = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return full_text, new_text


def main() -> None:
    args = parse_args()
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)
    prompt = load_prompt(args.input_file)
    tokenizer = prepare_tokenizer(args.model_path)

    runs = [
        ("local modeling_llama", LocalLlamaForCausalLM),
        ("transformers llama", HfLlamaForCausalLM),
    ]

    for name, model_cls in runs:
        print(f"=== {name} ===")
        model = load_model(model_cls, args.model_path, dtype, device)
        full_text, new_text = greedy_generate(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
        )
        if args.print_full:
            print(full_text)
        else:
            print(new_text)

        del model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

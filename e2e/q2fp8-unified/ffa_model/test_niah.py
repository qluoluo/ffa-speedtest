#!/usr/bin/env python
"""
Simple NIAH (Needle In A Haystack) test for FFA-Q2FP8-Sym model
直接测试，不依赖 opencompass 框架
"""
import os
import sys
import json
import argparse

# 直接添加模块路径
_BASE = '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym'
sys.path.insert(0, _BASE)

import torch
from transformers import AutoTokenizer, AutoConfig

def main():
    parser = argparse.ArgumentParser(description="NIAH Test for FFA-Q2FP8-Sym")
    parser.add_argument("--k-bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--use-ffa", action="store_true", default=True, help="Enable FFA decode")
    parser.add_argument("--no-ffa", action="store_true", help="Disable FFA decode (use flash_attn only)")
    parser.add_argument("--max-length", type=int, default=None, help="Truncate prompt to this length (tokens)")
    args = parser.parse_args()

    use_ffa = not args.no_ffa

    print("=" * 70)
    print(f"NIAH Test: FFA-Q{args.k_bits}FP8-Sym (use_ffa_decode={use_ffa}, delta={args.delta})")
    print("=" * 70)

    # Import model components
    print("\n[1] Importing modules...")
    from modeling_llama import LlamaForCausalLM
    from q2fp8_cache import Q2FP8SymCache
    print("Import successful!")

    # Load NIAH data
    print("\n[2] Loading NIAH data...")
    niah_path = "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/attn_analysis/data/NeedleBench/Length32000Depth42_origin_en_32k.json"
    with open(niah_path, 'r') as f:
        niah_data = json.load(f)

    item = niah_data['0']
    prompt = item['origin_prompt']
    gold = item['gold']

    print(f"Prompt length: {len(prompt)} chars")
    print(f"Gold answer: {gold}")

    # Load model
    MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
    print(f"\n[3] Loading model from: {MODEL_PATH}")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": use_ffa,
        "delta": args.delta,
        "BS": 128,
        "use_fp8_residual": True,
        "k_bits": args.k_bits,
    }

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded!")

    # Load tokenizer
    print("\n[4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Tokenize
    print("\n[5] Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length or 131072)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]
    print(f"Input token length: {input_len}")

    # Create cache
    print("\n[6] Creating Q2FP8SymCache...")
    past_kv = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=args.k_bits)

    # Generate
    print("\n[7] Generating response...")
    print("-" * 50)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            past_key_values=past_kv,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Question: What legendary item is hidden on Emerald Island?")
    print(f"Expected: {gold}")
    print(f"Generated: {generated_text}")
    print()

    # Check if answer is correct
    gold_keywords = ['Magic Essence']
    is_correct = any(kw.lower() in generated_text.lower() for kw in gold_keywords)

    if is_correct:
        print("✓ CORRECT! The model found the needle.")
    else:
        print("✗ INCORRECT. The model failed to find the needle.")

    print("=" * 70)

    # Print cache stats
    if past_kv.layers:
        layer0 = past_kv.layers[0]
        print(f"\nCache stats (layer 0):")
        print(f"  Quantized blocks: {layer0.num_full_blocks}")
        print(f"  Quantized tokens: {layer0.get_quantized_len()}")
        print(f"  Current (unquantized) tokens: {layer0.get_current_len()}")
        print(f"  Total seq length: {layer0.get_seq_length()}")

if __name__ == "__main__":
    main()

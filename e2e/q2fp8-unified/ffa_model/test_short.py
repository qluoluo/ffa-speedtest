#!/usr/bin/env python
"""
简单测试：用短序列验证 FFA-Q2FP8-Sym 和原版 transformers 输出一致
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache

def test_short_generation():
    print("=" * 70)
    print("Testing FFA-Q2FP8-Sym vs Original Transformers (short sequence)")
    print("=" * 70)

    MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

    # 简单的测试 prompt
    test_prompts = [
        "The capital of France is",
        "Hello, my name is",
        "1 + 1 =",
    ]

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    print("\n[2] Loading original transformers model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model_orig.eval()

    # Load FFA model
    print("\n[3] Loading FFA-Q2FP8-Sym model...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": True,
        "delta": 5.0,
        "BS": 128,
        "use_fp8_residual": True,
        "k_bits": 2,
    }
    model_ffa = FFALlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model_ffa.eval()

    print("\n[4] Testing generation...")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs['input_ids'].shape[1]

        # Original model
        with torch.no_grad():
            outputs_orig = model_orig.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text_orig = tokenizer.decode(outputs_orig[0][input_len:], skip_special_tokens=True)

        # FFA model (with flash_attn, no FFA decode for short sequences)
        config.attn_settings["use_ffa_decode"] = False
        with torch.no_grad():
            outputs_ffa_flash = model_ffa.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text_ffa_flash = tokenizer.decode(outputs_ffa_flash[0][input_len:], skip_special_tokens=True)

        # FFA model with Q2FP8 cache
        config.attn_settings["use_ffa_decode"] = True
        cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)
        with torch.no_grad():
            outputs_ffa_q2 = model_ffa.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=cache,
            )
        text_ffa_q2 = tokenizer.decode(outputs_ffa_q2[0][input_len:], skip_special_tokens=True)

        print(f"  Original:     '{text_orig}'")
        print(f"  FFA (flash):  '{text_ffa_flash}'")
        print(f"  FFA (Q2FP8):  '{text_ffa_q2}'")

        # Check if outputs match
        if text_orig == text_ffa_flash == text_ffa_q2:
            print("  ✓ All outputs match!")
        else:
            print("  ⚠ Outputs differ!")

    print("\n" + "=" * 70)
    print("Test complete!")

if __name__ == "__main__":
    test_short_generation()

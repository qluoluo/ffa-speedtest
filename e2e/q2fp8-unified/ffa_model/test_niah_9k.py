#!/usr/bin/env python
"""
9K 长度的大海捞针测试：使用真实评测数据比较 FFA-Q2FP8-Sym 和原版 transformers 的输出
数据来源: Length9000Depth42_origin_en_32k.json
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache


def main():
    print("=" * 70)
    print("NIAH Test (9K): FFA-Q2FP8-Sym vs Original Transformers")
    print("=" * 70)

    MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
    DATA_PATH = "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/eval_script/ffa-q2fp8-sym/oc-eval-result/niah/20260116_171435/predictions/ffa-q2fp8-sym-delta5/Length9000Depth42_origin_en_32k.json"

    # 加载测试数据
    print("\n[1] Loading test data...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    item = data['0']
    prompt = item['origin_prompt']
    gold = item['gold']

    print(f"Prompt length: {len(prompt)} chars")
    print(f"Gold answer: {gold}")

    # 找到 needle 位置
    needle_pos = prompt.find("Hidden on Emerald Island")
    print(f"Needle position: char {needle_pos} ({needle_pos/len(prompt)*100:.1f}%)")

    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize 看看长度
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs['input_ids'].shape[1]
    print(f"Input tokens: {input_len}")

    # Load original model
    print("\n[3] Loading original transformers model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model_orig.eval()

    # Load FFA model
    print("\n[4] Loading FFA-Q2FP8-Sym model...")
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

    print("\n[5] Running inference...")
    print("=" * 70)

    inputs = inputs.to("cuda")

    # Original model
    print("\nOriginal model generating...")
    with torch.no_grad():
        outputs_orig = model_orig.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text_orig = tokenizer.decode(outputs_orig[0][input_len:], skip_special_tokens=True)

    # FFA model
    print("FFA model generating...")
    cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)
    with torch.no_grad():
        outputs_ffa = model_ffa.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            past_key_values=cache,
        )
    text_ffa = tokenizer.decode(outputs_ffa[0][input_len:], skip_special_tokens=True)

    # 检查结果
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nGold answer: {gold}")
    print(f"\nOriginal output: '{text_orig.strip()}'")
    print(f"FFA output:      '{text_ffa.strip()}'")

    # 检查是否包含正确答案
    correct_orig = "Magic Essence" in text_orig
    correct_ffa = "Magic Essence" in text_ffa
    match = text_orig.strip() == text_ffa.strip()

    print(f"\nOriginal correct: {'YES' if correct_orig else 'NO'}")
    print(f"FFA correct:      {'YES' if correct_ffa else 'NO'}")
    print(f"Outputs match:    {'YES' if match else 'NO'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
用原版 transformers 测试 NIAH（作为基准）
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 70)
    print("NIAH Test: Original Transformers (baseline)")
    print("=" * 70)

    # Load NIAH data
    print("\n[1] Loading NIAH data...")
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
    print(f"\n[2] Loading model from: {MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded!")

    # Load tokenizer
    print("\n[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Tokenize
    print("\n[4] Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=131072)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]
    print(f"Input token length: {input_len}")

    # Generate
    print("\n[5] Generating response...")
    print("-" * 50)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
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

if __name__ == "__main__":
    main()

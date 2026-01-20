#!/usr/bin/env python
"""
2K 长度的大海捞针测试：比较 FFA-Q2FP8-Sym 和原版 transformers 的输出
"""
import sys
sys.path.insert(0, '/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa_q2fp8_sym')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache


def create_niah_prompt(needle: str, target_tokens: int = 2000, needle_position: float = 0.5, tokenizer=None):
    """
    创建大海捞针测试 prompt。

    Args:
        needle: 要隐藏的关键信息
        target_tokens: 目标 token 数量
        needle_position: needle 在文本中的位置 (0.0 = 开头, 1.0 = 结尾)
        tokenizer: 用于计算 token 数
    """
    # 填充文本（无关内容）
    filler_sentences = [
        "The weather today is quite pleasant with clear skies and mild temperatures.",
        "Scientists continue to explore the mysteries of the universe through advanced telescopes.",
        "The global economy shows signs of gradual recovery after recent challenges.",
        "Technology companies are investing heavily in artificial intelligence research.",
        "Environmental conservation efforts are gaining momentum worldwide.",
        "The art exhibition features works from various contemporary artists.",
        "Medical researchers announce promising results in cancer treatment studies.",
        "Urban planning experts discuss sustainable city development strategies.",
        "The music festival attracted thousands of visitors from different countries.",
        "Educational institutions are adapting to new digital learning methods.",
        "Sports events bring communities together and promote healthy lifestyles.",
        "The culinary world embraces fusion cuisine combining different traditions.",
        "Space agencies plan ambitious missions to explore distant planets.",
        "Financial markets respond to changes in monetary policy decisions.",
        "Cultural heritage sites receive protection under new preservation laws.",
        "The film industry continues to evolve with streaming platforms.",
        "Agricultural practices are becoming more sustainable and efficient.",
        "Public transportation systems are being modernized in many cities.",
        "The fashion industry is embracing more eco-friendly materials.",
        "Wildlife conservation programs help protect endangered species.",
    ]

    # 构建填充文本直到达到目标 token 数
    question = f"\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer: The secret code is"

    filler_text = ""
    i = 0
    while True:
        filler_text += filler_sentences[i % len(filler_sentences)] + " "
        i += 1
        # 检查当前长度
        test_prompt = filler_text + f" {needle} " + filler_text[:100] + question
        if tokenizer and len(tokenizer.encode(test_prompt)) >= target_tokens:
            break
        if i > 10000:  # 防止无限循环
            break

    # 计算 needle 插入位置
    insert_pos = int(len(filler_text) * needle_position)

    # 插入 needle
    text_before = filler_text[:insert_pos]
    text_after = filler_text[insert_pos:]

    # 构建完整 prompt
    full_context = text_before + f" {needle} " + text_after

    # 调整长度以达到目标 token 数
    full_prompt = full_context + question
    if tokenizer:
        tokens = tokenizer.encode(full_prompt)
        while len(tokens) > target_tokens and len(full_context) > 100:
            full_context = full_context[:-50]
            full_prompt = full_context + question
            tokens = tokenizer.encode(full_prompt)

    return full_prompt


def main():
    print("=" * 70)
    print("NIAH Test (2K): FFA-Q2FP8-Sym vs Original Transformers")
    print("=" * 70)

    MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

    # 创建测试数据
    needle = "IMPORTANT: The secret code is ALPHA-7749-OMEGA. Remember this code."

    # 测试不同位置的 needle
    test_cases = [
        ("Beginning (10%)", 0.1),
        ("Middle (50%)", 0.5),
        ("End (90%)", 0.9),
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

    print("\n[4] Running tests...")
    print("=" * 70)

    results = []

    for test_name, needle_pos in test_cases:
        print(f"\n--- Test: Needle at {test_name} ---")

        # 创建 prompt
        prompt = create_niah_prompt(
            needle,
            target_tokens=2000,
            needle_position=needle_pos,
            tokenizer=tokenizer
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs['input_ids'].shape[1]
        print(f"Input length: {input_len} tokens")

        # Original model
        with torch.no_grad():
            outputs_orig = model_orig.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text_orig = tokenizer.decode(outputs_orig[0][input_len:], skip_special_tokens=True)

        # FFA model
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
        correct_orig = "ALPHA-7749-OMEGA" in text_orig
        correct_ffa = "ALPHA-7749-OMEGA" in text_ffa
        match = text_orig.strip() == text_ffa.strip()

        print(f"Original:  '{text_orig.strip()[:100]}...'")
        print(f"FFA:       '{text_ffa.strip()[:100]}...'")
        print(f"Original correct: {'✓' if correct_orig else '✗'}")
        print(f"FFA correct:      {'✓' if correct_ffa else '✗'}")
        print(f"Outputs match:    {'✓' if match else '✗'}")

        results.append({
            "test": test_name,
            "orig_correct": correct_orig,
            "ffa_correct": correct_ffa,
            "match": match,
            "orig_output": text_orig.strip(),
            "ffa_output": text_ffa.strip(),
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<20} {'Original':<12} {'FFA':<12} {'Match':<10}")
    print("-" * 54)
    for r in results:
        orig_status = '✓ Correct' if r['orig_correct'] else '✗ Wrong'
        ffa_status = '✓ Correct' if r['ffa_correct'] else '✗ Wrong'
        match_status = '✓' if r['match'] else '✗'
        print(f"{r['test']:<20} {orig_status:<12} {ffa_status:<12} {match_status:<10}")

    print("=" * 70)

    # Overall stats
    orig_correct_count = sum(1 for r in results if r['orig_correct'])
    ffa_correct_count = sum(1 for r in results if r['ffa_correct'])
    match_count = sum(1 for r in results if r['match'])

    print(f"\nOriginal model accuracy: {orig_correct_count}/{len(results)}")
    print(f"FFA model accuracy:      {ffa_correct_count}/{len(results)}")
    print(f"Output match rate:       {match_count}/{len(results)}")


if __name__ == "__main__":
    main()

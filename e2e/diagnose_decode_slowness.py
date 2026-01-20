#!/usr/bin/env python3
"""
诊断 E2E decode 慢的原因
分析每个 decode step 的详细耗时
"""

import sys
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified" / "ffa_model"))
sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified"))

# Compatibility patch
import transformers.integrations
if not hasattr(transformers.integrations, 'use_kernel_forward_from_hub'):
    from compat_patch import use_kernel_forward_from_hub
    transformers.integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub

from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache

MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

def profile_single_decode_step(model, input_ids, cache, attention_mask=None):
    """Profile a single decode step with detailed timing"""

    timings = {}

    # Total time
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    timings['total'] = (t_end - t_start) * 1000

    return outputs, timings

def main():
    device = torch.device('cuda:0')
    dtype = torch.float16

    print("="*80)
    print("E2E Decode Performance Diagnosis")
    print("="*80)

    # Load model
    print("\nLoading Q2FP8-Unified model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.attn_settings = {
        "use_ffa_decode": True,
        "delta": 5.0,
        "BS": 128,
        "SBS": 128,
        "use_fp8_residual": True,
        "k_bits": 2,
    }

    model = FFALlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Test with different prompt lengths
    for prompt_len in [8192, 32768]:
        print(f"\n{'='*80}")
        print(f"Testing with prompt length: {prompt_len}")
        print(f"{'='*80}")

        # Generate prompt
        base_text = "The quick brown fox jumps over the lazy dog. " * 10
        prompt = base_text * (prompt_len // len(tokenizer.encode(base_text)) + 1)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_len).to(device)
        actual_len = inputs["input_ids"].shape[1]
        print(f"Actual prompt length: {actual_len}")

        # Create cache
        cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)

        # Prefill phase
        print("\n--- Prefill Phase ---")
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                past_key_values=cache,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        prefill_time = (t_end - t_start) * 1000

        print(f"Prefill time: {prefill_time:.2f} ms")
        print(f"Prefill throughput: {actual_len / (prefill_time / 1000):.2f} tok/s")

        # Decode phase - profile first 10 tokens
        print("\n--- Decode Phase (first 10 tokens) ---")

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        attention_mask = inputs.get("attention_mask")

        decode_times = []
        for step in range(10):
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            step_time = (t_end - t_start) * 1000
            decode_times.append(step_time)

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            print(f"  Step {step+1}: {step_time:.2f} ms")

        avg_decode = sum(decode_times) / len(decode_times)
        print(f"\nAverage decode time: {avg_decode:.2f} ms")
        print(f"Decode throughput: {1000/avg_decode:.2f} tok/s")

        # Check cache statistics
        print("\n--- Cache Statistics ---")
        for layer_idx in [0, 15, 31]:  # Check first, middle, last layer
            layer_cache = cache.layers[layer_idx]
            if layer_cache is not None:
                quantized_len = layer_cache.get_quantized_len()
                current_len = layer_cache.get_current_len()
                print(f"Layer {layer_idx}: quantized={quantized_len}, current={current_len}")

        del cache, past_key_values
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Diagnosis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

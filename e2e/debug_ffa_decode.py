#!/usr/bin/env python3
"""
调试脚本：检查 FFA decode kernel 是否被正确启用
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified" / "ffa_model"))

# Compatibility patch
import transformers.integrations
if not hasattr(transformers.integrations, 'use_kernel_forward_from_hub'):
    sys.path.insert(0, str(Path(__file__).parent / "q2fp8-unified"))
    from compat_patch import use_kernel_forward_from_hub
    transformers.integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub

from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
from q2fp8_cache import Q2FP8SymCache

MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
DEVICE = "cuda:0"

def main():
    print("=" * 80)
    print("FFA Decode Debug Script")
    print("=" * 80)

    # Load model
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

    print(f"\nLoading model with config:")
    print(f"  use_ffa_decode: {config.attn_settings['use_ffa_decode']}")
    print(f"  delta: {config.attn_settings['delta']}")
    print(f"  BS: {config.attn_settings['BS']}")

    model = FFALlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Create cache
    cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)

    # Test 1: Prefill phase (should use flash_attn)
    print("\n" + "=" * 80)
    print("Test 1: Prefill Phase (32768 tokens)")
    print("=" * 80)

    prompt = "The quick brown fox jumps over the lazy dog. " * 2000
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768).to(DEVICE)
    actual_len = inputs["input_ids"].shape[1]
    print(f"Actual prompt length: {actual_len}")

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            past_key_values=cache,
            use_cache=True,
        )

    # Check cache state
    print(f"\nCache state after prefill:")
    print(f"  Total seq length: {cache.get_seq_length()}")
    print(f"  Quantized length: {cache.get_quantized_len()}")
    print(f"  Current buffer length: {cache.get_current_len()}")
    print(f"  Number of layers: {len(cache.layers)}")

    if len(cache.layers) > 0:
        layer0 = cache.layers[0]
        print(f"\nLayer 0 details:")
        print(f"  k_q shape: {layer0.k_q.shape if layer0.k_q is not None else None}")
        print(f"  k_scale shape: {layer0.k_scale.shape if layer0.k_scale is not None else None}")
        print(f"  k_residual shape: {layer0.k_residual.shape if layer0.k_residual is not None else None}")
        print(f"  k_current shape: {layer0.k_current.shape if layer0.k_current is not None else None}")
        print(f"  v_current shape: {layer0.v_current.shape if layer0.v_current is not None else None}")
        print(f"  num_full_blocks: {layer0.num_full_blocks}")
        print(f"  current_len: {layer0.current_len}")

        # Check if quantized blocks exist
        has_quantized = layer0.k_q is not None and layer0.k_scale is not None
        print(f"  Has quantized blocks: {has_quantized}")

    # Test 2: Decode phase (should use FFA if conditions met)
    print("\n" + "=" * 80)
    print("Test 2: Decode Phase (first token)")
    print("=" * 80)

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    print(f"Next token shape: {next_token.shape}")

    # Add instrumentation to check FFA path
    print("\nChecking FFA decode conditions:")
    print(f"  q_len == 1: {next_token.shape[1] == 1}")
    print(f"  use_ffa_decode: {config.attn_settings['use_ffa_decode']}")
    print(f"  is_q2fp8_cache: {isinstance(cache, Q2FP8SymCache)}")

    if len(cache.layers) > 0:
        layer0 = cache.layers[0]
        has_quantized_blocks = layer0.k_q is not None and layer0.k_scale is not None
        print(f"  has_quantized_blocks: {has_quantized_blocks}")
        print(f"  layer_idx in pattern_layers: True (default all layers)")

    # Run decode step
    with torch.no_grad():
        decode_outputs = model(
            input_ids=next_token,
            attention_mask=torch.cat([
                inputs["attention_mask"],
                torch.ones((1, 1), device=DEVICE, dtype=inputs["attention_mask"].dtype)
            ], dim=1) if inputs.get("attention_mask") is not None else None,
            past_key_values=cache,
            use_cache=True,
        )

    print(f"\nDecode step completed successfully")
    print(f"Cache seq length after decode: {cache.get_seq_length()}")
    print(f"Quantized length: {cache.get_quantized_len()}")
    print(f"Current buffer length: {cache.get_current_len()}")

    # Test 3: Multiple decode steps
    print("\n" + "=" * 80)
    print("Test 3: Multiple Decode Steps (10 tokens)")
    print("=" * 80)

    past_kv = decode_outputs.past_key_values
    next_tok = decode_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    attention_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones((1, 1), device=DEVICE, dtype=inputs["attention_mask"].dtype)
    ], dim=1) if inputs.get("attention_mask") is not None else None

    for step in range(10):
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=DEVICE, dtype=attention_mask.dtype)
            ], dim=1)

        with torch.no_grad():
            step_outputs = model(
                input_ids=next_tok,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
            )

        past_kv = step_outputs.past_key_values
        next_tok = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if step % 5 == 0:
            print(f"  Step {step}: seq_len={past_kv.get_seq_length()}, "
                  f"quantized={past_kv.get_quantized_len()}, "
                  f"current={past_kv.get_current_len()}")

    print("\n" + "=" * 80)
    print("Debug Complete")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Final seq length: {past_kv.get_seq_length()}")
    print(f"  Final quantized length: {past_kv.get_quantized_len()}")
    print(f"  Final current buffer: {past_kv.get_current_len()}")

    if len(past_kv.layers) > 0:
        layer0 = past_kv.layers[0]
        has_quantized = layer0.k_q is not None and layer0.k_scale is not None
        print(f"  Has quantized blocks: {has_quantized}")

        if has_quantized:
            print(f"\n✓ FFA decode should be enabled (all conditions met)")
        else:
            print(f"\n✗ FFA decode NOT enabled (no quantized blocks)")

    print("\nIf FFA decode is enabled but still slow, the issue is likely:")
    print("  1. Current tokens processing overhead (nested loops)")
    print("  2. Quantization overhead during decode")
    print("  3. Kernel launch overhead (not using CUDA Graph)")

if __name__ == "__main__":
    main()

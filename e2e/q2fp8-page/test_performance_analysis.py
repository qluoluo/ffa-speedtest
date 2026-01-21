"""
Performance analysis script to identify bottlenecks.
"""
import torch
import time
from transformers import AutoTokenizer, AutoConfig
from ffa_model.modeling_llama import LlamaForCausalLM
from ffa_model.q2fp8_cache import Q2FP8SymCache

def test_decode_performance(use_ffa: bool, model_path: str, prefill_len: int = 16384, decode_tokens: int = 256):
    """Test decode performance with detailed timing."""

    # Load model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_settings = {
        'use_ffa_decode': use_ffa,
        'delta': 5.0,
        'BS': 128,
        'k_bits': 2,
        'use_fp8_residual': True,
        'return_skip_ratio': False,
        'debug_stats': {},
    }

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create input
    prompt = 'Hello ' * 4000
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=prefill_len)
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['attention_mask'].to('cuda')

    # Create cache
    cache = Q2FP8SymCache(BS=128, use_fp8_residual=True, k_bits=2)

    # Prefill
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=cache,
            logits_to_keep=1,
        )
    next_token = outputs.logits.argmax(dim=-1)
    if next_token.dim() == 1:
        next_token = next_token.unsqueeze(-1)

    # Align to BS boundary
    current_len = cache.get_seq_length() % 128
    if current_len > 0:
        align_tokens = 128 - current_len
        for _ in range(align_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=cache,
                    logits_to_keep=1,
                )
            next_token = outputs.logits.argmax(dim=-1)
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(-1)

    # Warmup
    for _ in range(4):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                use_cache=True,
                past_key_values=cache,
                logits_to_keep=1,
            )
        next_token = outputs.logits.argmax(dim=-1)
        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)

    # Timed decode
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(decode_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                use_cache=True,
                past_key_values=cache,
                logits_to_keep=1,
            )
        next_token = outputs.logits.argmax(dim=-1)
        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput = decode_tokens / elapsed

    # Get debug stats
    stats = model.config.attn_settings.get('debug_stats', {})

    # Cleanup
    del model
    del cache
    torch.cuda.empty_cache()

    return {
        'method': 'FFA-Q2FP8' if use_ffa else 'FlashAttention',
        'decode_time': elapsed,
        'throughput': throughput,
        'debug_stats': stats,
    }

if __name__ == '__main__':
    model_path = '/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B'

    print("="*70)
    print("Performance Analysis")
    print("="*70)

    # Test FlashAttention
    print("\nTesting FlashAttention...")
    flash_result = test_decode_performance(False, model_path, decode_tokens=256)
    print(f"FlashAttention: {flash_result['throughput']:.2f} tokens/sec")
    print(f"Debug stats: {flash_result['debug_stats']}")

    # Test FFA
    print("\nTesting FFA-Q2FP8...")
    ffa_result = test_decode_performance(True, model_path, decode_tokens=256)
    print(f"FFA-Q2FP8: {ffa_result['throughput']:.2f} tokens/sec")
    print(f"Debug stats: {ffa_result['debug_stats']}")

    # Compare
    speedup = ffa_result['throughput'] / flash_result['throughput']
    print(f"\nSpeedup: {speedup:.2f}x")

    # Analyze stats
    ffa_stats = ffa_result['debug_stats']
    if 'ffa_graph_replay' in ffa_stats and 'ffa_need_lse' in ffa_stats:
        print(f"\nGraph replay: {ffa_stats['ffa_graph_replay']}")
        print(f"Need LSE: {ffa_stats['ffa_need_lse']}")
        print(f"Graph recapture: {ffa_stats.get('ffa_graph_recapture', 0)}")
        print(f"Current len nonzero: {ffa_stats.get('ffa_current_len_nonzero', 0)}")

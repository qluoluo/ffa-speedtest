#!/usr/bin/env python3
"""
测试所有模型和kernel的导入是否正确
"""
import sys
from pathlib import Path

print("=" * 70)
print("Testing FFA-Q2FP8 Imports")
print("=" * 70)

# Add FFA paths
sys.path.insert(0, str(Path(__file__).parent / "q2fp8" / "ffa_model"))
sys.path.insert(0, str(Path(__file__).parent / "q2fp8" / "attn_kernel"))

try:
    from modeling_llama import LlamaForCausalLM as FFALlamaForCausalLM
    print("✓ FFA modeling_llama.LlamaForCausalLM imported successfully")
except Exception as e:
    print(f"✗ FFA modeling_llama import failed: {e}")

try:
    from q2fp8_cache import Q2FP8SymCache
    print("✓ FFA q2fp8_cache.Q2FP8SymCache imported successfully")
except Exception as e:
    print(f"✗ FFA q2fp8_cache import failed: {e}")

try:
    from ffa_fwd_decode import attn_forward_decode
    print("✓ FFA ffa_fwd_decode.attn_forward_decode imported successfully")
except Exception as e:
    print(f"✗ FFA ffa_fwd_decode import failed: {e}")

try:
    from attn_q2fp8_sym_mask import attn_forward_decode_quantized
    print("✓ FFA attn_q2fp8_sym_mask.attn_forward_decode_quantized imported successfully")
except Exception as e:
    print(f"✗ FFA attn_q2fp8_sym_mask import failed: {e}")

print("\n" + "=" * 70)
print("Testing Quest Imports")
print("=" * 70)

# Add Quest paths - need to add parent and rename quest_model to quest
quest_model_path = Path(__file__).parent / "quest" / "quest_model"
sys.path.insert(0, str(quest_model_path.parent))

# Create a symlink or import directly from models
try:
    # Try importing directly from the models submodule
    sys.path.insert(0, str(quest_model_path))
    from models.llama import LlamaForCausalLM as QuestLlama
    print("✓ Quest LlamaForCausalLM imported successfully (direct from models)")
except Exception as e:
    print(f"✗ Quest import failed: {e}")
    print(f"  Note: Quest requires 'transformers' package in conda environment")

print("\n" + "=" * 70)
print("Testing Shared Utilities")
print("=" * 70)

sys.path.insert(0, str(Path(__file__).parent / "shared"))

try:
    from benchmark_utils import BenchmarkResult, Timer
    print("✓ benchmark_utils imported successfully")
except Exception as e:
    print(f"✗ benchmark_utils import failed: {e}")

try:
    from test_prompts import get_test_prompts
    print("✓ test_prompts imported successfully")
except Exception as e:
    print(f"✗ test_prompts import failed: {e}")

print("\n" + "=" * 70)
print("All Import Tests Completed")
print("=" * 70)

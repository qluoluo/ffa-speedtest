"""
OpenCompass Model Registration for FFA-Q2FP8-Sym

对称量化 + Page-wise 策略 + 阈值筛选。
支持 2-bit 和 4-bit 量化。
"""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

from opencompass.models.myModel.hf_strip_model import (
    HuggingFaceCausalLM_Strip as HuggingFaceCausalLM,
)
from .modeling_llama import LlamaForCausalLM
from .q2fp8_cache import Q2FP8SymCache
from transformers import AutoConfig


@MODELS.register_module()
class HF_ForCausalLM_FFA_Q2FP8_Sym_OC(HuggingFaceCausalLM):
    """
    FFA-Q2FP8-Sym Model for OpenCompass Evaluation.

    对称量化 + Page-wise 策略：
    - 使用 2-bit 或 4-bit 对称量化 + FP8 残差存储 K cache
    - 按 block 独立量化，新增 token 满一个 block 才量化
    - 通过阈值筛选减少计算量
    """

    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        model_kwargs = kwargs
        attn_defaults = dict(
            use_ffa_decode=False,
            delta=5.0,
            pattern_layers=None,
            BS=128,
            SBS=None,
            return_skip_ratio=False,
            use_fp8_residual=True,
            k_bits=2,  # 量化位数: 2 或 4
            skip_ratio_store=None,
        )

        config_attn_settings = {key: model_kwargs.pop(key, default) for key, default in attn_defaults.items()}
        config_attn_settings = {k: v for k, v in config_attn_settings.items() if v is not None}

        trust_remote_code = model_kwargs.get("trust_remote_code", False)

        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", None)
        if model_type not in ("llama",):
            raise ValueError(
                "HF_ForCausalLM_FFA_Q2FP8_Sym_OC only supports llama models, "
                f"got model_type={model_type} for path={path}"
            )
        config.attn_settings = config_attn_settings
        self.config_attn_settings = config_attn_settings
        print(f"[DEBUG] FFA-Q2FP8-Sym OC Model config_attn_settings: {config_attn_settings}")

        if model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=path, config=config, **model_kwargs
            )

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        BS = self.config_attn_settings.get("BS", 128)
        use_fp8_residual = self.config_attn_settings.get("use_fp8_residual", True)
        k_bits = self.config_attn_settings.get("k_bits", 2)

        fresh_cache = Q2FP8SymCache(BS=BS, use_fp8_residual=use_fp8_residual, k_bits=k_bits)

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs["past_key_values"] = fresh_cache

        if "past_key_values" in kwargs:
            print(f"[DEBUG] WARNING: kwargs contains past_key_values, type={type(kwargs['past_key_values'])}")

        print(f"[DEBUG] generate() called, num_inputs={len(inputs)}, use_ffa_decode={self.config_attn_settings.get('use_ffa_decode')}, k_bits={k_bits}")
        print(f"[DEBUG] Input sample (first 100 chars): {inputs[0][:100] if inputs else 'N/A'}...")

        old_gen_kwargs = self.generation_kwargs
        self.generation_kwargs = gen_kwargs
        try:
            result = super().generate(inputs, **kwargs)
            print(f"[DEBUG] Output sample (first 100 chars): {result[0][:100] if result else 'N/A'}...")
            return result
        finally:
            self.generation_kwargs = old_gen_kwargs

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# from transformers.models.qwen2.modeling_qwen2 import (
#     Qwen2Attention,
#     apply_rotary_pos_emb,
#     repeat_kv
# )

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import load_from_babilong_json, load_from_longbench_jsonl


def modify_model_attn(model, save_dirpath: Path):
    """
    修改模型的前向传播以捕获注意力模式
    """
    save_dirpath = Path(save_dirpath)

    def custom_attn_forward(self, 
                            hidden_states: torch.Tensor,
                            position_embeddings: tuple[torch.Tensor, torch.Tensor],
                            *args, **kwargs):
        # print("Enter Attn custom forward")

        # 获取层索引
        layer_idx = self.layer_idx
        layer_save_dirpath = save_dirpath / f"layer_{layer_idx}"
        layer_save_dirpath.mkdir(parents=True, exist_ok=True)
        
        # 准备注意力计算
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # import ipdb; ipdb.set_trace()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        torch.save(hidden_states, layer_save_dirpath / "h.pt")
        torch.save(query_states, layer_save_dirpath / "q_unrope.pt")
        torch.save(key_states, layer_save_dirpath / "k_unrope.pt")
        torch.save(value_states, layer_save_dirpath / "v.pt")

        # 应用位置编码
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        torch.save(query_states, layer_save_dirpath / "q_rope.pt")
        torch.save(key_states, layer_save_dirpath / "k_rope.pt")

        print(f"Layer {layer_idx} saved qkvh for shape {query_states.shape=} {key_states.shape=} dtype={query_states.dtype}")

        # 处理键值重复
        # rep_nums = query_states.shape[1] // key_states.shape[1]
        # key_states = repeat_kv(key_states, rep_nums)

        return self._original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            *args,
            **kwargs,
        )

    # 修改所有层的注意力前向传播
    for layer in model.model.layers:
        self_attention = layer.self_attn
        self_attention._original_forward = self_attention.forward
        self_attention.forward = custom_attn_forward.__get__(self_attention, type(self_attention))

    return model


def parse_args():
    ffa_root = Path(__file__).resolve().parents[2]
    default_opencompass_root = ffa_root / "huffkv-opencompass"
    parser = argparse.ArgumentParser(description="Dump q/k/v/h tensors from a model forward pass.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"
        ),
        help="Path to the HF model directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional tokenizer path. Defaults to model path.",
    )
    parser.add_argument(
        "--opencompass-root",
        type=Path,
        default=None,
        help="Root path for huffkv-opencompass. Used to infer dataset defaults.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["longbench", "babilong"],
        default="longbench",
        help="Which dataset loader to use.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Explicit dataset path. Overrides the dataset-type default.",
    )
    parser.add_argument("--line-start", type=int, default=48)
    parser.add_argument("--line-end", type=int, default=68)
    parser.add_argument("--line-idx", type=int, default=50)
    parser.add_argument("--repeat-text-num", type=int, default=1)
    parser.add_argument("--sample-len-k", type=int, default=256)
    parser.add_argument(
        "--save-root",
        type=Path,
        default=None,
        help="Root folder to save results. Defaults to opencompass attn_analysis/result.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Model load dtype.",
    )
    parser.add_argument("--device-map", type=str, default="auto")

    args = parser.parse_args()

    opencompass_root = args.opencompass_root or default_opencompass_root
    if args.dataset_path is None:
        if args.dataset_type == "longbench":
            args.dataset_path = opencompass_root / "data/Longbench/data/gov_report.jsonl"
        else:
            args.dataset_path = opencompass_root / "data/babilong/data/qa1/16k.json"
    args.opencompass_root = opencompass_root
    args.save_root = args.save_root or (
        opencompass_root / "opencompass/models/myModel/ffa/attn_analysis/result"
    )
    return args


def map_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


if __name__ == "__main__":
    args = parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    if args.dataset_type == "longbench":
        raw_text, dataset_name = load_from_longbench_jsonl(
            str(args.dataset_path), args.line_start, args.line_end
        )
    else:
        raw_text, dataset_name = load_from_babilong_json(
            str(args.dataset_path), args.line_idx
        )

    if args.repeat_text_num > 1:
        raw_text = "\n".join([raw_text] * args.repeat_text_num)
        dataset_name = f"{dataset_name}_repeat{args.repeat_text_num}"

    input_ids = tokenizer(
        raw_text, truncation=False, padding=False, return_tensors="pt"
    ).input_ids

    sample_len = args.sample_len_k * 1024
    if args.sample_len_k > 0 and input_ids.shape[-1] >= sample_len:
        print(f"cut {input_ids.shape[-1]//1024}k to {args.sample_len_k}k length..")
        input_ids = input_ids[..., :sample_len]
        dataset_name = f"{dataset_name}_{args.sample_len_k}k"

    print(f"{input_ids.shape=}")

    save_dirpath = Path(args.save_root) / args.model_path.name / dataset_name
    save_dirpath.mkdir(parents=True, exist_ok=True)
    raw_text_savefp = save_dirpath / "raw_text.txt"
    raw_text_savefp.write_text(raw_text)
    save_layerdata_dirpath = save_dirpath / "layer_data"
    save_layerdata_dirpath.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=map_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = modify_model_attn(model, save_layerdata_dirpath)

    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        model(input_ids)

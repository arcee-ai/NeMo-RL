from collections import defaultdict
import json
import os
import torch
from nemo_rl.models.custom.llama3.model import Transformer
from nemo_rl.models.custom.llama3.args import TransformerModelArgs
from nemo_rl.models.custom.llama3.state_dict_adapter import Llama3StateDictAdapter

from transformers import AutoTokenizer, AutoModelForCausalLM

from safetensors.torch import load_file

def load_hf_state_dict(model_args, hf_assets_path):
    index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
    hf_state = {}

    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]  # param_name -> shard filename
        shard_to_keys = defaultdict(list)
        for k, shard in weight_map.items():
            shard_to_keys[shard].append(k)

        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(hf_assets_path, shard)
            shard_sd = load_file(shard_path, device="cpu")
            for k in keys:
                hf_state[k] = shard_sd[k]
    else:
        # single-file fallback
        single_path = os.path.join(hf_assets_path, "model.safetensors")
        hf_state = load_file(single_path, device="cpu")

    return hf_state

args_8b = TransformerModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
)

print("Init model and adapter")

model_tt = Transformer(model_args=args_8b)
adapter = Llama3StateDictAdapter(model_args=args_8b, hf_assets_path="Llama-3-8B")

print("Load HF state dict")
hf_state_dict = load_hf_state_dict(args_8b, "Llama-3-8B")

print("Convert HF state dict to TT state dict")
tt_state_dict = adapter.from_hf(hf_state_dict)

print("Load TT state dict into model")
model_tt.load_state_dict(tt_state_dict)

print("Evaluate model")
model_tt.eval()

tokenizer = AutoTokenizer.from_pretrained("Llama-3-8B")

prompt = "The first US president was George"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

logits_tt = model_tt(input_ids)
probs_tt = torch.softmax(logits_tt[:, -1, :], dim=-1)

top_token_ids = probs_tt.argmax(dim=-1)

top_token = tokenizer.decode(top_token_ids[0])

print(f"Next token: {top_token}")

print("Load HF model")
model_hf = AutoModelForCausalLM.from_pretrained("Llama-3-8B")
model_hf.eval()

print("Evaluate HF model")

logits_hf = model_hf(input_ids).logits

kldiv = torch.nn.functional.kl_div(logits_hf, logits_tt)

print(f"KL divergence: {kldiv}")
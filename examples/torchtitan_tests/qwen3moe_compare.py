from accelerate import init_empty_weights
import torch
import torch.nn.functional as F

from nemo_rl.models.custom.convert import get_model_config
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

config = AutoConfig.from_pretrained(model_name)

model_class, model_args, state_dict_adapter_class, parallelize_fn = get_model_config(config)

print("create tt model")
with init_empty_weights():
    model_tt = model_class(model_args)
state_dict_adapter = state_dict_adapter_class(model_args, hf_assets_path=model_name)

print("load hf model")
model_hf = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=("auto" if device == "cuda" else "cpu"),
    low_cpu_mem_usage=True,
    use_safetensors=True,
    torch_dtype=(torch.bfloat16 if device == "cuda" else None),
)

print("load state dict into tt")
state_dict_tt = state_dict_adapter.from_hf(model_hf.state_dict())
failed = False

import re
number_pattern = "\\.{[0-9]+:[0-9]+}\\."
seen = []
for key in model_tt.state_dict().keys():
    de_numbered = re.sub(number_pattern, "[N]", key)
    if key not in state_dict_tt.keys():
        if de_numbered not in seen:
            print(f"model_tt has key {key} but state_dict_tt does not")
            failed = True
        else:
            seen.append(de_numbered)

seen = []
for key in state_dict_tt.keys():
    de_numbered = re.sub(number_pattern, "[N]", key)
    if key not in model_tt.state_dict().keys():
        if de_numbered not in seen:
            print(f"state_dict_tt has key {key} but model_tt does not")
            failed = True
        else:
            seen.append(de_numbered)
if failed:
    raise ValueError("state_dict_tt and model_tt do not have the same keys")
model_tt.load_state_dict(state_dict_tt, assign=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model_hf.eval()
model_tt.eval()
model_tt.to(device)

tokenizer.padding_side = "left"

prompt = """Two generations now had pass'd away,
Wise by his rules, and happy by his sway;
Two ages o'er his native realm he reign'd, [011]
And now the example of the third remain'd.
All view'd with awe the venerable man;
Who thus with mild benevolence began:—
"What shame, what woe is this to Greece! what joy
To Troy's proud monarch, and the friends of Troy!
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=False)
input_ids = inputs["input_ids"].to(device)

print("run hf model")
with torch.inference_mode():
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_hf = model_hf(input_ids).logits
    else:
        logits_hf = model_hf(input_ids.cpu()).logits
print("run tt model")
with torch.inference_mode():
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_tt = model_tt(input_ids)
    else:
        logits_tt = model_tt(input_ids)

# Compare only the last time step and move minimal data to CPU
tt_last = logits_tt[:, -1, :].detach().to(torch.float32).to("cpu")
hf_last = logits_hf[:, -1, :].detach().to(torch.float32).to("cpu")

max_abs_diff = (tt_last - hf_last).abs().max()
mse = F.mse_loss(tt_last, hf_last)

kl_tt_hf = F.kl_div(
    F.log_softmax(tt_last, dim=-1),
    F.softmax(hf_last, dim=-1),
    reduction="batchmean",
)
kl_hf_tt = F.kl_div(
    F.log_softmax(hf_last, dim=-1),
    F.softmax(tt_last, dim=-1),
    reduction="batchmean",
)

print(f"max_abs_diff: {max_abs_diff.item():.6e}")
print(f"mse: {mse.item():.6e}")
print(f"KL(tt||hf): {kl_tt_hf.item():.6e}")
print(f"KL(hf||tt): {kl_hf_tt.item():.6e}")
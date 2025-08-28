from accelerate import init_empty_weights
import torch
import torch.nn.functional as F

from nemo_rl.models.custom.convert import get_model_config
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

config = AutoConfig.from_pretrained(model_name)

model_class, model_args, state_dict_adapter_class, parallelize_fn = get_model_config(config)

print("create tt model")
with init_empty_weights():
    model_tt = model_class(model_args)
state_dict_adapter = state_dict_adapter_class(model_args, hf_assets_path=model_name)

print("load hf model")
model_hf = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

print("load state dict into tt")
state_dict_tt = state_dict_adapter.from_hf(model_hf.state_dict())
assert state_dict_tt.keys() == model_tt.state_dict().keys()
model_tt.load_state_dict(state_dict_tt, assign=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_hf.eval()
model_tt.eval()
model_tt.to("cuda")

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

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]

print("run hf model")
logits_hf = model_hf(input_ids).logits
print("run tt model")
with torch.no_grad():
    logits_tt = model_tt(input_ids.to("cuda"))

tt = logits_tt.detach().to(torch.float32).to("cpu")
hf = logits_hf.detach().to(torch.float32).to("cpu")

B, T, V = tt.shape
tt_flat = tt.reshape(-1, V)
hf_flat = hf.reshape(-1, V)

max_abs_diff = (tt_flat - hf_flat).abs().max()
mse = F.mse_loss(tt_flat, hf_flat)

kl_tt_hf = F.kl_div(
    F.log_softmax(tt_flat, dim=-1),
    F.softmax(hf_flat, dim=-1),
    reduction="batchmean",
)
kl_hf_tt = F.kl_div(
    F.log_softmax(hf_flat, dim=-1),
    F.softmax(tt_flat, dim=-1),
    reduction="batchmean",
)

print(f"max_abs_diff: {max_abs_diff.item():.6e}")
print(f"mse: {mse.item():.6e}")
print(f"KL(tt||hf): {kl_tt_hf.item():.6e}")
print(f"KL(hf||tt): {kl_hf_tt.item():.6e}")
"""Manually compare our custom implementation of a model with the native RLKit implementation."""
import sys

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rlkit.models.convert import get_model_config

model_name = sys.argv[1]

device = "cuda"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

model_class, model_args, state_dict_adapter_class = get_model_config(config)

print("create tt model")
with init_empty_weights():
    model_tt = model_class(model_args)
state_dict_adapter = state_dict_adapter_class(model_args, hf_assets_path=model_name)

print("load hf model")
model_hf = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
    use_safetensors=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

print("load state dict into tt")
state_dict_tt = state_dict_adapter.from_hf(model_hf.state_dict())
failed = False

check_keys_granular = False

if check_keys_granular:
    import re
    number_pattern = "\\.[0-9]+\\."
    seen = []
    for key in model_tt.state_dict().keys():
        de_numbered = re.sub(number_pattern, ".[N].", key)
        if key not in state_dict_tt.keys():
            if de_numbered not in seen:
                print(f"model_tt has key {de_numbered} but state_dict_tt does not")
                failed = True
                seen.append(de_numbered)

    seen = []
    for key in state_dict_tt.keys():
        de_numbered = re.sub(number_pattern, ".[N].", key)
        if key not in model_tt.state_dict().keys():
            if de_numbered not in seen:
                print(f"state_dict_tt has key {de_numbered} but model_tt does not")
                failed = True
                seen.append(de_numbered)
    if failed:
        raise ValueError("state_dict_tt and model_tt do not have the same keys")
else:
    assert state_dict_tt.keys() == model_tt.state_dict().keys()
model_tt.load_state_dict(state_dict_tt, assign=True, strict=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_hf.eval()
model_tt.eval()
model_tt.to(device)

tokenizer.padding_side = "left"

prompt = prompt = """Two generations now had pass'd away,
Wise by his rules, and happy by his sway;
Two ages o'er his native realm he reign'd, [011]
And now the example of the third remain'd.
All view'd with awe the venerable man;
Who thus with mild benevolence began:—
"What shame, what woe is this to Greece! what joy
To Troy's proud monarch, and the friends of Troy!
That adverse gods commit to stern debate
The best, the bravest, of the Grecian state.
Young as ye are, this youthful heat restrain,
# A godlike race of heroes once I knew,
# Such as no more these aged eyes shall view!
# Lives there a chief to match Pirithous' fame,
# Dryas the bold, or Ceneus' deathless name;
# Theseus, endued with more than mortal might,
# Or Polyphemus, like the gods in fight?
# With these of old, to toils of battle bred,
# In early youth my hardy days I led;
# Fired with the thirst which virtuous envy breeds,
# And smit with love of honourable deeds,
# Strongest of men, they pierced the mountain boar,
# Ranged the wild deserts red with monsters' gore,
# And from their hills the shaggy Centaurs tore:
# Yet these with soft persuasive arts I sway'd;
# When Nestor spoke, they listen'd and obey'd.
# If in my youth, even these esteem'd me wise;
# Do you, young warriors, hear my age advise.
# Atrides, seize not on the beauteous slave;
# That prize the Greeks by common suffrage gave:
# Nor thou, Achilles, treat our prince with pride;
# Let kings be just, and sovereign power preside.
# 20 The Iliad of Homer
# Thee, the first honours of the war adorn,
# Like gods in strength, and of a goddess born;
# Him, awful majesty exalts above
# The powers of earth, and sceptred sons of Jove.
# Let both unite with well-consenting mind,
# So shall authority with strength be join'd.
# Leave me, O king! to calm Achilles' rage;
# Rule thou thyself, as more advanced in age.
# Forbid it, gods! Achilles should be lost,
# The pride of Greece, and bulwark of our host."
# This said, he ceased. The king of men replies:
# "Thy years are awful, and thy words are wise.
# But that imperious, that unconquer'd soul,
# No laws can limit, no respect control.
# Before his pride must his superiors fall;
# His word the law, and he the lord of all?
# Him must our hosts, our chiefs, ourself obey?
# What king can bear a rival in his sway?
# Grant that the gods his matchless force have given;
# Has foul reproach a privilege from heaven?"""

with torch.no_grad():
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
                logits_tt = model_tt(input_ids, attention_masks=model_tt.get_attention_masks(input_ids, tokenizer))
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

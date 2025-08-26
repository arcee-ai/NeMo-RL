# Simple test script to compare the output of a TT-like model with a HF model and validate equivalence.

from collections import defaultdict
import json
import os
import torch
import torch.nn.functional as F
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

prompt = """Two generations now had pass'd away,
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
Nor think your Nestor's years and wisdom vain.
A godlike race of heroes once I knew,
Such as no more these aged eyes shall view!
Lives there a chief to match Pirithous' fame,
Dryas the bold, or Ceneus' deathless name;
Theseus, endued with more than mortal might,
Or Polyphemus, like the gods in fight?
With these of old, to toils of battle bred,
In early youth my hardy days I led;
Fired with the thirst which virtuous envy breeds,
And smit with love of honourable deeds,
Strongest of men, they pierced the mountain boar,
Ranged the wild deserts red with monsters' gore,
And from their hills the shaggy Centaurs tore:
Yet these with soft persuasive arts I sway'd;
When Nestor spoke, they listen'd and obey'd.
If in my youth, even these esteem'd me wise;
Do you, young warriors, hear my age advise.
Atrides, seize not on the beauteous slave;
That prize the Greeks by common suffrage gave:
Nor thou, Achilles, treat our prince with pride;
Let kings be just, and sovereign power preside.
20 The Iliad of Homer
Thee, the first honours of the war adorn,
Like gods in strength, and of a goddess born;
Him, awful majesty exalts above
The powers of earth, and sceptred sons of Jove.
Let both unite with well-consenting mind,
So shall authority with strength be join'd.
Leave me, O king! to calm Achilles' rage;
Rule thou thyself, as more advanced in age.
Forbid it, gods! Achilles should be lost,
The pride of Greece, and bulwark of our host."
This said, he ceased. The king of men replies:
"Thy years are awful, and thy words are wise.
But that imperious, that unconquer'd soul,
No laws can limit, no respect control.
Before his pride must his superiors fall;
His word the law, and he the lord of all?
Him must our hosts, our chiefs, ourself obey?
What king can bear a rival in his sway?
Grant that the gods his matchless force have given;
Has foul reproach a privilege from heaven?"""

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

with torch.no_grad():
    logits_hf = model_hf(input_ids).logits

tt = logits_tt.detach().to(torch.float32)
hf = logits_hf.detach().to(torch.float32)

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
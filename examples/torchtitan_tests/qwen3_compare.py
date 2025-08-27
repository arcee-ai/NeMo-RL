# Simple test script to compare the output of a TT-like model with a HF model and validate equivalence.

from collections import defaultdict
import json
import os
import torch
import torch.nn.functional as F
from nemo_rl.models.custom.qwen3.model import Transformer
from nemo_rl.models.custom.qwen3.args import TransformerModelArgs
from nemo_rl.models.custom.qwen3.state_dict_adapter import Qwen3StateDictAdapter

from transformers import AutoTokenizer, AutoModelForCausalLM

from nemo_rl.models.custom.attention import init_attention_mask

args_8b = TransformerModelArgs(
    dim=4096,
    n_layers=36,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=151936,
    norm_eps=1e-6,
    rope_theta=1_000_000,
    head_dim=128,
    intermediate_size=12288,
    max_seq_len=40960,
    use_flex_attn=True,
    attn_mask_type="causal",
    attention_dropout=0.0,
    use_sliding_window=False,
    sliding_window=4096,   # ignored when use_sliding_window=False
    max_window_layers=36,
    eos_id=151645,
)

print("Load HF model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model_hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model_hf.eval()

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

dummy_batch = torch.empty(1, input_ids.size(1), 1, 1, dtype=torch.int32)
init_attention_mask(batch=dummy_batch, eos_id=tokenizer.eos_token_id)

print("Init model and adapter")

model_tt = Transformer(model_args=args_8b)
adapter = Qwen3StateDictAdapter(model_args=args_8b, hf_assets_path="Qwen/Qwen3-8B")

print("Convert HF state dict to TT state dict")
tt_state_dict = adapter.from_hf(model_hf.state_dict())

print("Load TT state dict into model")
model_tt.load_state_dict(tt_state_dict)

print("Evaluate model")
model_tt.eval()

logits_tt = model_tt(input_ids)
probs_tt = torch.softmax(logits_tt[:, -1, :], dim=-1)

top_token_ids = probs_tt.argmax(dim=-1)

top_token = tokenizer.decode(top_token_ids[0])

print(f"Next token: {top_token}")



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
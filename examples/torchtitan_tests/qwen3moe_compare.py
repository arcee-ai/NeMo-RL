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
model_hf = AutoModelForCausalLM.from_pretrained(model_name)

print("load state dict into tt")
model_tt.load_state_dict(state_dict_adapter.from_hf(model_hf.state_dict()), assign=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_hf.eval()
model_tt.eval()

tokenizer.padding_side = "left"

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

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]

print("run tt model")
logits_tt = model_tt(input_ids)
print("run hf model")
logits_hf = model_hf(inputs)

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
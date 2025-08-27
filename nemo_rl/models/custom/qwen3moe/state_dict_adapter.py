import re
from typing import Any

import torch

from nemo_rl.models.custom.state_dict_adapter import StateDictAdapter


class Qwen3MoeStateDictAdapter(StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # Dense MLP path when a layer is non-MoE
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        self.to_hf_dense_map = {v: k for k, v in self.from_hf_map.items()}

        self._re_layer = re.compile(r"model\.layers\.(\d+)\.")
        self._re_dense_mlp = re.compile(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight")
        self._re_moe_router = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight")
        self._re_moe_expert = re.compile(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            # Split grouped experts to per-expert weights
            m = re.match(r"layers\.(\d+)\.moe\.experts\.(w[123])$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                assert isinstance(value, torch.Tensor) and value.dim() == 3
                for expert_idx in range(value.shape[0]):
                    if which == "w1":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                    elif which == "w3":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                    else:
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                    hf_state_dict[hf_key] = value[expert_idx]
                continue

            # Router gate mapping
            m = re.match(r"layers\.(\d+)\.moe\.router\.gate\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"] = value
                continue

            # Dense path and common weights via direct map
            if key in self.to_hf_dense_map:
                hf_key = self.to_hf_dense_map[key]
                hf_state_dict[hf_key] = value
                continue

            # Layer-indexed common weights
            if key.startswith("layers."):
                # Replace first number with {}
                abstract_key = re.sub(r"(layers\.)\d+", r"\1{}", key, count=1)
                if abstract_key in self.to_hf_dense_map:
                    m2 = re.search(r"layers\.(\d+)", key)
                    if m2 is None:
                        continue
                    layer_idx = m2.group(1)
                    hf_key = self.to_hf_dense_map[abstract_key].format(layer_idx)
                    hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        experts_accumulator: dict[str, dict[int, torch.Tensor]] = {}

        for key, value in hf_state_dict.items():
            # MoE experts per-expert weights → grouped tensors
            m = self._re_moe_expert.match(key)
            if m is not None:
                layer_idx = m.group(1)
                expert_idx = int(m.group(2))
                which = m.group(3)
                if which == "gate_proj":
                    native_key = f"layers.{layer_idx}.moe.experts.w1"
                elif which == "up_proj":
                    native_key = f"layers.{layer_idx}.moe.experts.w3"
                else:
                    native_key = f"layers.{layer_idx}.moe.experts.w2"
                bucket = experts_accumulator.setdefault(native_key, {})
                bucket[expert_idx] = value
                continue

            # MoE router gate
            m = self._re_moe_router.match(key)
            if m is not None:
                layer_idx = m.group(1)
                state_dict[f"layers.{layer_idx}.moe.router.gate.weight"] = value
                continue

            # Dense MLP path
            m = self._re_dense_mlp.match(key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                if which == "gate_proj":
                    native_key = f"layers.{layer_idx}.feed_forward.w1.weight"
                elif which == "up_proj":
                    native_key = f"layers.{layer_idx}.feed_forward.w3.weight"
                else:
                    native_key = f"layers.{layer_idx}.feed_forward.w2.weight"
                state_dict[native_key] = value
                continue

            # Common mappings with layer index
            if "layers" in key:
                abstract_key = re.sub(r"(model\.layers\.)\d+", r"\1{}", key, count=1)
                m2 = re.search(r"model\.layers\.(\d+)", key)
                if m2 is None:
                    continue
                layer_num = m2.group(1)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is not None:
                    state_dict[new_key.format(layer_num)] = value
                continue

            # Non-layered common weights
            new_key = self.from_hf_map.get(key)
            if new_key is not None:
                state_dict[new_key] = value

        # Assemble grouped expert tensors
        for native_key, slices in experts_accumulator.items():
            max_idx = max(slices.keys())
            ordered = [slices[i] for i in range(max_idx + 1)]
            state_dict[native_key] = torch.stack(ordered, dim=0)

        return state_dict


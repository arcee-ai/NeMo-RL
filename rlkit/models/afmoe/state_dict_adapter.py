"""State dict adapter for AFMoE models."""
import re
from typing import Any

import torch

from rlkit.models.state_dict_adapter import BaseStateDictAdapter


class AFMoEStateDictAdapter(BaseStateDictAdapter):
    """State dict adapter for AFMoE models. Converts weight names as well as restructuring weights for efficient MoE support."""

    def __init__(self, model_args, hf_assets_path: str | None):
        """Initialize the state dict adapter."""
        super().__init__(model_args, hf_assets_path)

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention projections
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # Attention norms
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Attention gate
            "model.layers.{}.self_attn.gate_proj.weight": "layers.{}.attention.gate_proj.weight",
            # Dual attention normalization
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm_a.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.attention_norm_b.weight",
            # Dual FFN normalization
            "model.layers.{}.pre_mlp_layernorm.weight": "layers.{}.ffn_norm_a.weight",
            "model.layers.{}.post_mlp_layernorm.weight": "layers.{}.ffn_norm_b.weight",
            # Dense MLP path when a layer is non-MoE
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Final norm and output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        self.to_hf_dense_map = {v: k for k, v in self.from_hf_map.items()}

        self._re_layer = re.compile(r"model\.layers\.(\d+)\.")
        self._re_dense_mlp = re.compile(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight")
        self._re_moe_router = re.compile(r"model\.layers\.(\d+)\.mlp\.router\.gate\.weight")
        self._re_moe_expert = re.compile(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        )
        self._re_moe_shared_expert = re.compile(
            r"model\.layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight"
        )
        self._re_expert_bias = re.compile(r"model\.layers\.(\d+)\.mlp\.expert_bias")

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
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
                    else:  # w2
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                    hf_state_dict[hf_key] = value[expert_idx]
                continue

            # Router gate mapping
            m = re.match(r"layers\.(\d+)\.moe\.router\.gate\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_state_dict[f"model.layers.{layer_idx}.mlp.router.gate.weight"] = value
                continue

            # Expert bias mapping
            m = re.match(r"layers\.(\d+)\.moe\.expert_bias$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_state_dict[f"model.layers.{layer_idx}.mlp.expert_bias"] = value
                continue

            # Shared experts mapping (if exists)
            m = re.match(r"layers\.(\d+)\.moe\.shared_experts\.(w[123])\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                if which == "w1":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
                elif which == "w3":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
                else:  # w2
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
                hf_state_dict[hf_key] = value
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

    def get_hf_metadata(self, state_dict: dict[str, Any]) -> dict[str, tuple[Any, Any]]:  # type: ignore[override]
        """Return HF metadata without materializing per-expert tensors."""
        metadata: dict[str, tuple[Any, Any]] = {}

        for key, value in state_dict.items():
            m = re.match(r"layers\.(\d+)\.moe\.experts\.(w[123])$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                num_experts = value.shape[0]
                per_expert_shape = value.shape[1:]
                for expert_idx in range(num_experts):
                    if which == "w1":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                    elif which == "w3":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                    else:
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                    metadata[hf_key] = (per_expert_shape, value.dtype)
                continue

            m = re.match(r"layers\.(\d+)\.moe\.router\.gate\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_key = f"model.layers.{layer_idx}.mlp.router.gate.weight"
                metadata[hf_key] = (value.shape, value.dtype)
                continue

            m = re.match(r"layers\.(\d+)\.moe\.expert_bias$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_key = f"model.layers.{layer_idx}.mlp.expert_bias"
                metadata[hf_key] = (value.shape, value.dtype)
                continue

            m = re.match(r"layers\.(\d+)\.moe\.shared_experts\.(w[123])\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                if which == "w1":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
                elif which == "w3":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
                else:
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
                metadata[hf_key] = (value.shape, value.dtype)
                continue

            if key in self.to_hf_dense_map:
                hf_key = self.to_hf_dense_map[key]
                metadata[hf_key] = (value.shape, value.dtype)
                continue

            if key.startswith("layers."):
                abstract_key = re.sub(r"(layers\.)\d+", r"\1{}", key, count=1)
                if abstract_key in self.to_hf_dense_map:
                    m2 = re.search(r"layers\.(\d+)", key)
                    if m2 is None:
                        continue
                    layer_idx = m2.group(1)
                    hf_key = self.to_hf_dense_map[abstract_key].format(layer_idx)
                    metadata[hf_key] = (value.shape, value.dtype)

        return metadata

    def stream_hf_metadata(self, state_dict: dict[str, Any]):
        """Yield HF metadata without materializing per-expert tensors."""
        for key, value in state_dict.items():
            m = re.match(r"layers\.(\d+)\.moe\.experts\.(w[123])$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                num_experts = value.shape[0]
                per_expert_shape = value.shape[1:]
                for expert_idx in range(num_experts):
                    if which == "w1":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                    elif which == "w3":
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                    else:
                        hf_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                    yield hf_key, per_expert_shape, value.dtype
                continue

            m = re.match(r"layers\.(\d+)\.moe\.router\.gate\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_key = f"model.layers.{layer_idx}.mlp.router.gate.weight"
                yield hf_key, value.shape, value.dtype
                continue

            m = re.match(r"layers\.(\d+)\.moe\.expert_bias$", key)
            if m is not None:
                layer_idx = m.group(1)
                hf_key = f"model.layers.{layer_idx}.mlp.expert_bias"
                yield hf_key, value.shape, value.dtype
                continue

            m = re.match(r"layers\.(\d+)\.moe\.shared_experts\.(w[123])\.weight$", key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                if which == "w1":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
                elif which == "w3":
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
                else:
                    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
                yield hf_key, value.shape, value.dtype
                continue

            if key in self.to_hf_dense_map:
                hf_key = self.to_hf_dense_map[key]
                yield hf_key, value.shape, value.dtype
                continue

            if key.startswith("layers."):
                abstract_key = re.sub(r"(layers\.)\d+", r"\1{}", key, count=1)
                if abstract_key in self.to_hf_dense_map:
                    m2 = re.search(r"layers\.(\d+)", key)
                    if m2 is None:
                        continue
                    layer_idx = m2.group(1)
                    hf_key = self.to_hf_dense_map[abstract_key].format(layer_idx)
                    yield hf_key, value.shape, value.dtype

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from HuggingFace state dict to native model format."""
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
                else:  # down_proj
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

            # Expert bias
            m = self._re_expert_bias.match(key)
            if m is not None:
                layer_idx = m.group(1)
                state_dict[f"layers.{layer_idx}.moe.expert_bias"] = value
                continue

            # Shared experts (if exists)
            m = self._re_moe_shared_expert.match(key)
            if m is not None:
                layer_idx = m.group(1)
                which = m.group(2)
                if which == "gate_proj":
                    native_key = f"layers.{layer_idx}.moe.shared_experts.w1.weight"
                elif which == "up_proj":
                    native_key = f"layers.{layer_idx}.moe.shared_experts.w3.weight"
                else:  # down_proj
                    native_key = f"layers.{layer_idx}.moe.shared_experts.w2.weight"
                state_dict[native_key] = value
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
                else:  # down_proj
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

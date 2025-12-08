"""State dict adapter for a Qwen3 model."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

import logging
import re
from typing import Any

from rlkit.models.state_dict_adapter import BaseStateDictAdapter

from .args import Qwen3ModelArgs


class Qwen3StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter for Qwen3 model."""
    def __init__(self, model_args: Qwen3ModelArgs, hf_assets_path: str | None):
        """Initialize the state dict adapter."""
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # Map from HF -> local
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # per-head q/k RMSNorm
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        # No permutation needed for Qwen3 RoPE in this layout
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict: dict[str, Any] = {}
        for key, value in state_dict.items():
            # Strip name mangling from torch.compile
            key = key.replace("_orig_mod.", "") # noqa: PLW2901
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num_search = re.search(r"\d+", key)
                assert layer_num_search is not None, f"Layer number not found in key: {key}"
                layer_num = layer_num_search.group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    logging.warning(f"Key {key} not found in to_hf_map. Skipping.")
                    hf_state_dict[key] = value
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    logging.warning(f"Key {key} not found in to_hf_map. Skipping.")
                    continue
            hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from HuggingFace state dict to native model format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num_search = re.search(r"\d+", key)
                assert layer_num_search is not None, f"Layer number not found in key: {key}"
                layer_num = layer_num_search.group(0)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    logging.warning(f"Key {key} not found in from_hf_map. Skipping.")
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    logging.warning(f"Key {key} not found in from_hf_map. Skipping.")
                    continue
            state_dict[new_key] = value
        return state_dict



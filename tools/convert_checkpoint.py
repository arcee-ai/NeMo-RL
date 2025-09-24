#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Local imports
from nemo_rl.models.custom.convert import get_model_config
from nemo_rl.utils.native_checkpoint import load_checkpoint


def _resolve_dcp_path(dcp_path: str) -> str:
    """Resolve a step directory to the actual DCP checkpoint directory if needed.

    If dcp_path contains a 'policy' subdirectory, assume that's the DCP weights dir.
    Otherwise, return the path as-is.
    """
    dcp_path = os.path.abspath(dcp_path)
    policy_subdir = os.path.join(dcp_path, "policy")
    if os.path.isdir(policy_subdir):
        return policy_subdir
    return dcp_path


def _load_native_model_and_state(dcp_path: str, hf_model_name: str) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    """Instantiate TorchTitan-native model on meta and load DCP state.

    Returns the instantiated model, its full state dict (CPU tensors), and the HF config.
    """
    # Always construct config in float32 – weights are mastered in fp32 and sharded by DCP
    hf_config = AutoConfig.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model_class, model_args, adapter_class, _ = get_model_config(hf_config)

    # Build empty model on meta to avoid allocating huge tensors pre-load
    with init_empty_weights():
        model = model_class(model_args=model_args)

    # Load DCP into the model
    load_checkpoint(model=model, weights_path=os.path.abspath(dcp_path))

    # Collect a CPU state_dict with real tensors
    state_dict: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.detach().cpu()
        else:
            state_dict[k] = v

    # Return model and hf_config (needed for saving config)
    return model, state_dict, hf_config


def convert_dcp_to_hf_cli(dcp_path: str, hf_model_name: str, output_dir: str) -> None:
    dcp_path = _resolve_dcp_path(dcp_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(dcp_path):
        raise FileNotFoundError(f"DCP checkpoint not found: {dcp_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load native model and state
    print("Loading native model and state")
    _, native_state, hf_config = _load_native_model_and_state(dcp_path, hf_model_name)

    # Build adapter to map native -> HF keys
    print("Building adapter")
    _, model_args, adapter_class, _ = get_model_config(hf_config)
    adapter = adapter_class(model_args=model_args, hf_assets_path=hf_model_name)

    # Convert keys to HF format
    hf_state = adapter.to_hf(native_state)

    # Instantiate HF model and load state dict
    print("Loading HF model")
    hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading HF state dict: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading HF state dict: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

    print("Saving HF model")

    # Save full HF checkpoint
    hf_model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer (from provided HF model name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved HuggingFace checkpoint to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TorchTitan (DCP) checkpoint to HuggingFace format")
    parser.add_argument("dcp_path", type=str, help="Path to Torch DCP checkpoint directory or step dir (e.g., checkpoints/step_62 or checkpoints/step_62/policy)")
    parser.add_argument("hf_model_name", type=str, help="HuggingFace model name or path used for config/tokenizer (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("output_dir", type=str, help="Output directory to write HuggingFace checkpoint")

    args = parser.parse_args()

    convert_dcp_to_hf_cli(args.dcp_path, args.hf_model_name, args.output_dir)


if __name__ == "__main__":
    main()

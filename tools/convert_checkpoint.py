#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer

# Local imports
from nemo_rl.models.custom.convert import get_model_config
from nemo_rl.utils.native_checkpoint import load_checkpoint


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
    dcp_path = os.path.abspath(dcp_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(dcp_path):
        raise FileNotFoundError(f"DCP checkpoint not found: {dcp_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load native model and state
    model, native_state, hf_config = _load_native_model_and_state(dcp_path, hf_model_name)

    # Build adapter to map native -> HF keys
    _, model_args, adapter_class, _ = get_model_config(hf_config)
    adapter = adapter_class(model_args=model_args, hf_assets_path=hf_model_name)

    # Convert keys to HF format
    hf_state = adapter.to_hf(native_state)

    # Save weights
    weights_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(hf_state, weights_path)

    # Save config
    hf_config.save_pretrained(output_dir)

    # Save tokenizer (from provided HF model name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved HuggingFace checkpoint to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TorchTitan (DCP) checkpoint to HuggingFace format")
    parser.add_argument("dcp_path", type=str, help="Path to Torch DCP checkpoint directory (e.g., checkpoints/step_62/policy)")
    parser.add_argument("hf_model_name", type=str, help="HuggingFace model name or path used for config/tokenizer (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("output_dir", type=str, help="Output directory to write HuggingFace checkpoint")

    args = parser.parse_args()

    try:
        convert_dcp_to_hf_cli(args.dcp_path, args.hf_model_name, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

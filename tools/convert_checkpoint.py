#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Local imports
from rlkit.models.custom.convert import get_model_config
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def _resolve_dcp_path(dcp_path: str) -> str:
    """Resolve a step directory to the actual DCP checkpoint directory if needed.

    If dcp_path contains a 'policy' subdirectory, assume that's the DCP weights dir.
    Otherwise, return the path as-is.
    """
    dcp_path = os.path.abspath(dcp_path)
    policy_subdir = os.path.join(dcp_path, "policy", "weights")
    if os.path.isdir(policy_subdir):
        return policy_subdir
    return dcp_path


def _load_native_model_and_state(dcp_path: str, hf_model_name: str) -> tuple[dict[str, Any], Any]:
    """Load native TorchTitan state dict from a DCP checkpoint without dist init.

    Uses dcp_to_torch_save to materialize the checkpoint to a temporary torch file, then
    extracts the native model state dict under the 'model' key.

    Returns (native_state_dict, hf_config).
    """
    # Load HF config (float32 master weights)
    hf_config = AutoConfig.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Convert DCP checkpoint to a single torch file, then load and extract 'model'
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_weights = os.path.join(tmpdir, "weights.pt")
        dcp_to_torch_save(dcp_path, tmp_weights)
        bundle = torch.load(tmp_weights, map_location="cpu")

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError(
            "Unexpected checkpoint bundle format. Expected a dict with a top-level 'model' key."
        )

    native_state: dict[str, Any] = bundle["model"]
    return native_state, hf_config


def convert_dcp_to_hf_cli(dcp_path: str, hf_model_name: str, output_dir: str, push_to_hub: bool = False) -> None:
    dcp_path = _resolve_dcp_path(dcp_path)
    
    if not os.path.exists(dcp_path):
        raise FileNotFoundError(f"DCP checkpoint not found: {dcp_path}")

    if not push_to_hub:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use output_dir as the repo name (extract basename if it's a path)
        repo_name = os.path.basename(output_dir.rstrip('/'))

    # Load native model and state
    print("Loading native model and state")
    native_state, hf_config = _load_native_model_and_state(dcp_path, hf_model_name)

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

    # Save tokenizer (from provided HF model name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    if push_to_hub:
        print(f"Pushing model to HuggingFace Hub as private repository: {repo_name}")
        hf_model.push_to_hub(repo_name, private=True, safe_serialization=True)
        tokenizer.push_to_hub(repo_name, private=True)
        print(f"Successfully pushed model to HuggingFace Hub: {repo_name}")
    else:
        print("Saving HF model")
        # Save full HF checkpoint
        hf_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved HuggingFace checkpoint to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TorchTitan (DCP) checkpoint to HuggingFace format")
    parser.add_argument("dcp_path", type=str, help="Path to Torch DCP checkpoint directory or step dir (e.g., checkpoints/step_62 or checkpoints/step_62/policy)")
    parser.add_argument("hf_model_name", type=str, help="HuggingFace model name or path used for config/tokenizer (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("output_dir", type=str, help="Output directory to write HuggingFace checkpoint, or repo name if using --push-to-hub")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HuggingFace Hub as a private repository instead of saving to disk")

    args = parser.parse_args()

    convert_dcp_to_hf_cli(args.dcp_path, args.hf_model_name, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()

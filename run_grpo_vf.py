"""Entrypoint for RL training."""

import os

from rlkit.config.rl.vllm import HttpVllmConfig

# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import argparse
import pprint
import asyncio

from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch

from rlkit.algorithms.grpo import GRPOTrainer
from rlkit.distributed.virtual_cluster import init_ray
from rlkit.utils.config import load_config, parse_hydra_overrides
from rlkit.utils.logger import get_next_experiment_dir
from rlkit.config import RLConfig

import logging

# Cope with asyncio spamming console on certain crashes
logging.getLogger("asyncio").setLevel(logging.ERROR)
# Suppress httpx logging (HTTP request spam)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Convenience resolver for config files.
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

def configure_generation_config(
    config: HttpVllmConfig, tokenizer: PreTrainedTokenizerBase, is_eval: bool = False
) -> HttpVllmConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    config["pad_token_id"] = tokenizer.pad_token_id
    if config.get("stop_token_ids") is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # Set skip_tokenizer_init for the HTTP backend
    should_init_tokenizer = is_eval or config.get("stop_strings") is not None
    config["vllm_cfg"]["skip_tokenizer_init"] = not should_init_tokenizer

    return config

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides

def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        raise ValueError("A config file is required. Please specify a config file using the --config argument.")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: RLConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    if not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"
        logging.warning("Detected that P2P via shared memory is not available. Setting NCCL_SHM_DISABLE to 1.")
        if not config["checkpointing"].get("hf_checkpoint", False):
            raise ValueError("Running on a system configuration with bugged DCP checkpointing. Please set `checkpointing.hf_checkpoint` to `True` to use centralized HuggingFace checkpoints.")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["policy"]["model_name"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    trainer = GRPOTrainer(config)
    
    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()

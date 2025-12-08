"""Entrypoint for RL training."""

import os

# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import argparse
import asyncio
import logging
import pprint

import torch
import yaml

from rlkit.algorithms.grpo import GRPOTrainer
from rlkit.config.rl import RLConfig
from rlkit.distributed.virtual_cluster import init_ray
from rlkit.utils.logger import get_next_experiment_dir

# Cope with asyncio spamming console on certain crashes
logging.getLogger("asyncio").setLevel(logging.ERROR)
# Suppress httpx logging (HTTP request spam)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args = parser.parse_args()

    return args

def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    if not args.config:
        raise ValueError("A config file is required. Please specify a config file using the --config argument.")

    with open(args.config) as f:
        config_unstructured = yaml.load(f, Loader=yaml.FullLoader)

    config = RLConfig.model_validate(config_unstructured)

    # Print config
    print("Final config:")
    pprint.pprint(config.model_dump())

    # Inexplicably, NCCL cannot cope with P2P via shared memory on some machines.
    if not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"

    # Get the next experiment directory with incremented ID
    config.logging.log_dir = get_next_experiment_dir(config.logging.log_dir)
    print(f"📊 Using log directory: {config.logging.log_dir}")
    if config.checkpointing.enabled:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing.checkpoint_dir}"
        )

    init_ray()

    trainer = GRPOTrainer(config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()

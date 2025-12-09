"""Entrypoint for SFT training."""

import os

from rich.logging import RichHandler

# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import argparse
import asyncio
import logging
import pprint

import torch
import yaml

from rlkit.algorithms.sft import SFTTrainer
from rlkit.config.sft import SFTConfig
from rlkit.distributed.virtual_cluster import init_ray
from rlkit.utils.logger import get_next_experiment_dir

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Cope with asyncio spamming console on certain crashes
logging.getLogger("asyncio").setLevel(logging.ERROR)
# Suppress httpx logging (HTTP request spam)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "config", type=str, help="Path to YAML config file"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    with open(args.config) as f:
        config_unstructured = yaml.load(f, Loader=yaml.FullLoader)

    config = SFTConfig.model_validate(config_unstructured)

    # Print config
    print("Final config:")
    pprint.pprint(config.model_dump())

    # Handle NCCL P2P issues
    if not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"

    # Get the next experiment directory with incremented ID
    config.logging.log_dir = get_next_experiment_dir(config.logging.log_dir)
    print(f"📊 Using log directory: {config.logging.log_dir}")
    if config.checkpointing.enabled:
        print(f"📊 Using checkpoint directory: {config.checkpointing.checkpoint_dir}")

    init_ray()

    trainer = SFTTrainer(config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()

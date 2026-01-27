"""CLI entrypoints for RLKit training runs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pprint
from collections.abc import Mapping
from typing import Any

import torch
import yaml
from rich.logging import RichHandler

from rlkit.distributed.virtual_cluster import init_ray
from rlkit.utils.logger import get_next_experiment_dir


def _configure_environment(*, enable_color_prefix: bool) -> None:
    """Set environment variables needed for Ray + uv runs."""
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    if enable_color_prefix:
        os.environ["RAY_COLOR_PREFIX"] = "1"


def _configure_logging(level: str, handler_kwargs: Mapping[str, Any]) -> None:
    """Configure the root logger with Rich formatting."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(**handler_kwargs)],
    )
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load_config(path: str, config_cls) -> Any:
    """Load YAML config and validate with the supplied Pydantic model."""
    with open(path) as f:
        config_unstructured = yaml.load(f, Loader=yaml.FullLoader)
    return config_cls.model_validate(config_unstructured)


def _maybe_disable_nccl_shm() -> None:
    """Disable NCCL SHM when P2P is unavailable."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"


def _prepare_log_dirs(config) -> None:
    """Assign log and checkpoint directories."""
    config.logging.log_dir = get_next_experiment_dir(config.logging.log_dir)
    print(f"📊 Using log directory: {config.logging.log_dir}")
    if config.checkpointing.enabled:
        print(f"📊 Using checkpoint directory: {config.checkpointing.checkpoint_dir}")


def _run_training(
    *,
    config_path: str,
    config_cls,
    trainer_cls,
    log_level: str,
    rich_kwargs: Mapping[str, Any],
    enable_color_prefix: bool,
) -> None:
    _configure_environment(enable_color_prefix=enable_color_prefix)
    _configure_logging(log_level, rich_kwargs)

    config = _load_config(config_path, config_cls)

    print("Final config:")
    pprint.pprint(config.model_dump())

    _maybe_disable_nccl_shm()
    _prepare_log_dirs(config)
    init_ray()

    trainer = trainer_cls(config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    asyncio.run(trainer.train())


def _parse_config_arg(description: str) -> str:
    """Parse a single positional config argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    return args.config


def rl() -> None:
    """Run GRPO training with an RL config."""
    from rlkit.algorithms.grpo import GRPOTrainer
    from rlkit.config.rl import RLConfig

    config_path = _parse_config_arg("Run GRPO training with configuration")
    _run_training(
        config_path=config_path,
        config_cls=RLConfig,
        trainer_cls=GRPOTrainer,
        log_level="INFO",
        rich_kwargs={
            "rich_tracebacks": True,
            "show_time": True,
            "show_level": False,
            "show_path": True,
            "markup": True,
        },
        enable_color_prefix=True,
    )


def sft() -> None:
    """Run SFT training with an SFT config."""
    from rlkit.algorithms.sft import SFTTrainer
    from rlkit.config.sft import SFTConfig

    config_path = _parse_config_arg("Run SFT training with configuration")
    _run_training(
        config_path=config_path,
        config_cls=SFTConfig,
        trainer_cls=SFTTrainer,
        log_level="INFO",
        rich_kwargs={"rich_tracebacks": True},
        enable_color_prefix=False,
    )

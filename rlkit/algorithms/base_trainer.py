"""Base trainer class with shared functionality for SFT and RL trainers."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rlkit.algorithms.utils import set_seed
from rlkit.config.base import BaseConfig
from rlkit.config.checkpointing import CheckpointingConfig
from rlkit.config.policy import PolicyConfig
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.training.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import Timer

logger = logging.getLogger(__name__)

# Generic type for save state - each trainer defines its own TypedDict
SaveStateT = TypeVar("SaveStateT")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xd Yh Zm' or shorter if less than a day/hour.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string like '7d 12h 45m' or '2h 30m' or '15m'.
    """
    if seconds < 0:
        return "0m"

    total_minutes = int(seconds / 60)
    minutes = total_minutes % 60
    total_hours = total_minutes // 60
    hours = total_hours % 24
    days = total_hours // 24

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")

    return " ".join(parts)


class SamplesPerSecondEMA:
    """Exponential moving average tracker for samples per second."""

    def __init__(self, alpha: float = 0.1):
        """Initialize the EMA tracker.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher values give more weight to recent samples.
        """
        self.alpha = alpha
        self.ema: float | None = None

    def update(self, samples: int, elapsed_seconds: float) -> float:
        """Update the EMA with a new measurement.

        Args:
            samples: Number of samples processed.
            elapsed_seconds: Time taken to process the samples.

        Returns:
            The updated EMA value.
        """
        if elapsed_seconds <= 0:
            return self.ema or 0.0

        current_rate = samples / elapsed_seconds
        if self.ema is None:
            self.ema = current_rate
        else:
            self.ema = self.alpha * current_rate + (1 - self.alpha) * self.ema
        return self.ema

    def get(self) -> float:
        """Get the current EMA value."""
        return self.ema or 0.0


class BaseTrainer[SaveStateT](ABC):
    """Base class for all trainers (SFT, GRPO, etc.).

    Provides shared functionality for:
    - Logger setup
    - Checkpointing setup and management
    - Cluster setup
    - Policy initialization
    - Sequence packing and distribution
    """

    # Shared attributes set by subclasses
    policy: Policy
    logger: Logger
    checkpointer: CheckpointManager
    tokenizer: PreTrainedTokenizerBase
    save_state: SaveStateT


    @abstractmethod
    def _get_default_save_state(self) -> SaveStateT:
        """Return the default save state for this trainer type."""
        ...

    @abstractmethod
    async def train(self) -> None:
        """Main training loop."""
        ...

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the base trainer."""
        self.config = config
        self.policy_config = config.policy
        self.logging_config = config.logging
        self.checkpointing_config = config.checkpointing

        self.logger = Logger(config.category, config.name, self.logging_config)
        self.logger.log_hyperparams(self.config.model_dump())

        self.checkpointer, self.save_state, last_checkpoint_path = self._setup_checkpointing(self.checkpointing_config)
        self.train_cluster = self._setup_train_cluster(self.policy_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_config.model_name)

        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None

        self.policy = Policy(
            cluster=self.train_cluster,
            config=self.policy_config,
            tokenizer=self.tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
        )

        set_seed(self.config.seed)

    def _get_pad_values(self) -> dict[str, int | float | bool]:
        """Return pad values for sequence packing."""
        return {
            "token_ids": self.tokenizer.pad_token_id,
            "token_mask": False,
            "advantages": 0.0,
            "generation_logprobs": -9999.0,
        }

    def _setup_checkpointing(
        self, checkpointing_config: CheckpointingConfig
    ) -> tuple[CheckpointManager, SaveStateT, str | None]:
        """Set up checkpointing - same pattern, different state types."""
        checkpointer = CheckpointManager(checkpointing_config)
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        save_state = cast(
            SaveStateT | None,
            checkpointer.load_training_info(last_checkpoint_path),
        )
        if save_state is None:
            save_state = self._get_default_save_state()
        return checkpointer, save_state, last_checkpoint_path

    def _setup_train_cluster(self, policy_config: PolicyConfig) -> RayVirtualCluster:
        """Create a Ray virtual cluster for training."""
        train_resources = policy_config.training.resources
        gpus_per_node = train_resources.gpus_per_node
        num_nodes = train_resources.num_nodes

        cluster = RayVirtualCluster(
            name="train_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logger.info(f"Training cluster initialized with {num_nodes} nodes and {gpus_per_node} GPUs per node.")
        return cluster

    def _save_checkpoint(self, step: int, timer: Timer) -> None:
        """Save a training checkpoint.

        Saves self.save_state (which should be updated by the training loop
        before calling this method) along with model weights and optimizer state.

        Args:
            step: Current training step (0-indexed, will be saved as step+1).
            timer: Timer instance for timing the checkpoint.
        """
        self.policy.prepare_for_training()

        with timer.time("checkpointing"):
            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                step + 1,
                cast(dict[str, Any], self.save_state),
                self.config.model_dump(),
            )
            self.policy.save_checkpoint(
                weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            )
            self.checkpointer.finalize_checkpoint(checkpoint_path)

    def _get_bin_stats(self, bins: list[dict[str, list]]) -> dict[str, float]:
        """Get statistics about packed bins for logging."""
        bin_lengths = [len(bin["token_ids"]) for bin in bins]
        return {
            "min_bin_length": min(bin_lengths) if bin_lengths else 0,
            "max_bin_length": max(bin_lengths) if bin_lengths else 0,
            "mean_bin_length": sum(bin_lengths) / len(bin_lengths) if bin_lengths else 0,
            "num_bins": len(bins),
        }

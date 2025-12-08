"""Base trainer class with shared functionality for SFT and RL trainers."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, cast

from transformers import PreTrainedTokenizerBase

from rlkit.config.checkpointing import CheckpointingConfig
from rlkit.config.logging import LoggingConfig
from rlkit.config.policy import PolicyConfig
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.training.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import Timer

# Generic type for save state - each trainer defines its own TypedDict
SaveStateT = TypeVar("SaveStateT")


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

    # ─────────────────────────────────────────────────────────────────────────
    # ABSTRACT: Must be implemented by subclasses
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _get_default_save_state(self) -> SaveStateT:
        """Return the default save state for this trainer type."""
        ...

    @abstractmethod
    def _get_pad_values(self) -> dict[str, int | float | bool]:
        """Return pad values for sequence packing."""
        ...

    @abstractmethod
    def _get_config_for_logging(self) -> dict[str, Any]:
        """Return config dict for logging hyperparameters."""
        ...

    @abstractmethod
    async def train(self) -> None:
        """Main training loop - implemented differently for SFT vs RL."""
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED: Reusable implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_logger(self, logging_config: LoggingConfig) -> Logger:
        """Set up the logger - identical for all trainers."""
        logger = Logger(logging_config)
        logger.log_hyperparams(self._get_config_for_logging())
        return logger

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

    def _setup_train_cluster(
        self,
        num_nodes: int,
        gpus_per_node: int,
        name: str = "train_cluster",
    ) -> RayVirtualCluster:
        """Create a Ray virtual cluster for training."""
        cluster = RayVirtualCluster(
            name=name,
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"Ray cluster '{name}' initialized with {num_nodes} nodes, "
            f"{gpus_per_node} GPUs per node"
        )
        return cluster

    def _initialize_policy(
        self,
        cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        weights_path: Path | str | None = None,
        optimizer_path: Path | str | None = None,
        use_cut_cross_entropy: bool = False,
    ) -> Policy:
        """Initialize the policy - common pattern with optional extras."""
        return Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            use_cut_cross_entropy=use_cut_cross_entropy,
        )

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
                self._get_config_for_logging(),
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

from typing import TypedDict

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.data import DataConfig
from rlkit.config.logging import LoggerConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig


class RMConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int


class RMMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    rm: RMConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

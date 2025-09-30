from typing import TypedDict

from nemo_rl.config.rl.policy import PolicyConfig
from nemo_rl.config.data import DataConfig
from nemo_rl.config.logging import LoggerConfig
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.checkpointing import CheckpointingConfig


class SFTConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int


class SFTMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

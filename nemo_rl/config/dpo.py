from typing import TypedDict

from nemo_rl.config.rl.policy import PolicyConfig
from nemo_rl.config.data import DataConfig
from nemo_rl.config.logging import LoggerConfig
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.checkpointing import CheckpointingConfig


class DPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    preference_loss_weight: float
    sft_loss_weight: float


class DPOMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

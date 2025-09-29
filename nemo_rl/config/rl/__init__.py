from dataclasses import dataclass
from typing import Any

from nemo_rl.config.rl.policy import PolicyConfig
from nemo_rl.config.rl.loss import ClippedPGLossConfig
from nemo_rl.config.data import DataConfig
from nemo_rl.config.rl.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.checkpointing import CheckpointingConfig

@dataclass
class RLConfig:
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

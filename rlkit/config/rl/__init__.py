from typing import Any, TypedDict

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.rl.loss import ClippedPGLossConfig, CISPOLossConfig
from rlkit.config.data import DataConfig
from rlkit.config.rl.grpo import GRPOConfig, GRPOLoggerConfig, EnvironmentConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig

class RLConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig | CISPOLossConfig
    env: EnvironmentConfig
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
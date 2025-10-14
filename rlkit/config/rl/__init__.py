from typing import Any, TypedDict

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.rl.loss import ClippedPGLossConfig
from rlkit.config.data import DataConfig
from rlkit.config.rl.grpo import GRPOConfig, GRPOLoggerConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig

class RLConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

# Alias for backwards compatibility
GRPOMasterConfig = RLConfig

"""Base configuration for all training runs."""
from pydantic import BaseModel

from rlkit.config.checkpointing import CheckpointingConfig
from rlkit.config.logging import LoggingConfig
from rlkit.config.policy import PolicyConfig


class BaseConfig(BaseModel):
    """Base configuration for all training runs."""

    category: str
    name: str

    policy: PolicyConfig
    logging: LoggingConfig
    checkpointing: CheckpointingConfig

    seed: int = 42

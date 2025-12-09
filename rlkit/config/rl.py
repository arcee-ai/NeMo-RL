"""Root configuration for RL runs."""

from typing import Any

from pydantic import BaseModel

from rlkit.config.base import BaseConfig


class RolloutsConfig(BaseModel):
    """Configuration for rollouts."""

    # Number of rollouts to generate for each prompt.
    group_size: int
    # Maximum rollouts that can be generating at once. Set to None to use the training global batch size.
    max_concurrent_rollouts: int | None = None
    # Whether to leave out the current response when calculating the baseline.
    use_leave_one_out_baseline: bool = False
    # Whether to divide advantages by the standard deviation of the rewards.
    use_std_normalization: bool = False
    # Maximum number of steps out of date a rollout can be before being discarded and regenerated.
    max_staleness: int = 1


class EnvironmentConfig(BaseModel):
    """Configuration for a verifiers environment."""

    env_name: str
    # Passed into the verifiers load_environment function.
    env_kwargs: dict[str, Any]
    # How much of the context window a prompt can occupy before being filtered out.
    max_prompt_length_ratio: float = 1.0
    # Whether to shuffle the environment's dataset.
    shuffle: bool = False


class RLConfig(BaseConfig):
    """Root configuration for RL runs."""

    rollouts: RolloutsConfig
    env: EnvironmentConfig

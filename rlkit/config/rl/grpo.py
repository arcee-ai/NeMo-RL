"""Configuration for GRPO."""
from typing import Any, TypedDict, NotRequired

class GRPOConfig(TypedDict):
    """RL-specific configuration options."""
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    minibatch_advantage_renorm: NotRequired[bool]
    max_staleness: NotRequired[int]
    seed: int
    
    interleave_rollouts: bool
    skip_long_prompts: bool
    max_prompt_length_ratio: float
    
    run_vram_torture_test: bool  # Ignore environments and run all training on a full context window of nonsense.

class EnvironmentConfig(TypedDict):
    """Configuration for a verifiers environment."""
    env_name: str
    env_kwargs: dict[str, Any]
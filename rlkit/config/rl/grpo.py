from typing import Any, TypedDict, NotRequired

from rlkit.config.logging import LoggerConfig

class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int

class GRPOConfig(TypedDict):
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
    env_name: str
    env_kwargs: dict[str, Any]
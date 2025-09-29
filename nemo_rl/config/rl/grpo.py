from typing import TypedDict, NotRequired

from nemo_rl.config.logging import LoggerConfig

class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int

class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int
    seed: int
    max_rollout_turns: NotRequired[int | None]

"""Training worker configuration."""
from typing import TypedDict, NotRequired

class DTensorV2Config(TypedDict):
    """Configuration options for training workers."""
    enabled: bool
    activation_checkpointing: NotRequired[bool]
    tensor_parallel_size: NotRequired[int]
    context_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
    dp_replicate: NotRequired[int]
    expert_parallel_size: NotRequired[int]
    custom_parallel_plan: NotRequired[str]
    env_vars: NotRequired[dict[str, str]]

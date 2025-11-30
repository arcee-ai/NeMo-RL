from typing import TypedDict, NotRequired

class DTensorV2Config(TypedDict):
    enabled: bool
    activation_checkpointing: NotRequired[bool | None]
    tensor_parallel_size: NotRequired[int | None]
    context_parallel_size: NotRequired[int | None]
    pipeline_parallel_size: NotRequired[int | None]
    dp_replicate: NotRequired[int | None]
    expert_parallel_size: NotRequired[int | None]
    custom_parallel_plan: NotRequired[str | None]
    env_vars: NotRequired[dict[str, str] | None]

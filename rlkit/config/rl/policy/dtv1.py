from typing import TypedDict, NotRequired

class DTensorConfig(TypedDict):
    enabled: bool
    cpu_offload: NotRequired[bool | None]
    sequence_parallel: NotRequired[bool | None]
    activation_checkpointing: NotRequired[bool | None]
    tensor_parallel_size: NotRequired[int | None]
    context_parallel_size: NotRequired[int | None]
    custom_parallel_plan: NotRequired[str | None]

from typing import NotRequired, TypedDict

from nemo_rl.models.custom.model import BaseModelArgs

class DTensorV2Config(TypedDict):
    enabled: bool
    cpu_offload: NotRequired[bool]
    sequence_parallel: NotRequired[bool]
    activation_checkpointing: NotRequired[bool]
    tensor_parallel_size: NotRequired[int]
    context_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
    expert_parallel_size: NotRequired[int]
    custom_parallel_plan: NotRequired[str]
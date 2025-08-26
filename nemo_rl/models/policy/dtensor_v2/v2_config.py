from typing import NotRequired, TypedDict

from nemo_rl.models.custom.model import BaseModelArgs

class ModelConfig(TypedDict):
    adapter_cls: str
    model_args: BaseModelArgs
    hf_assets_path: str

class DTensorV2Config(TypedDict):
    enabled: bool
    model_config: ModelConfig
    cpu_offload: NotRequired[bool]
    sequence_parallel: NotRequired[bool]
    activation_checkpointing: NotRequired[bool]
    tensor_parallel_size: NotRequired[int]
    context_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
    expert_parallel_size: NotRequired[int]
    custom_parallel_plan: NotRequired[str]
from typing import NotRequired, TypedDict


class TorchTitanConfig(TypedDict):
    enabled: bool
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    env_vars: NotRequired[dict[str, str]]
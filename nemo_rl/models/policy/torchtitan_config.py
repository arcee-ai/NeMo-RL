from typing import NotRequired, TypedDict


class TorchTitanConfig(TypedDict):
    enabled: bool
    tensor_parallel_size: int
    context_parallel_size: int
    env_vars: NotRequired[dict[str, str]]
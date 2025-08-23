from typing import Any, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig

class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]
    enforce_eager: NotRequired[bool]
    extra_cli_args: NotRequired[list[str]]


class HttpVllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    gpus_per_node: int
    server_timeout: int
    num_nodes: NotRequired[int]
    vllm_kwargs: NotRequired[dict[str, Any]]
from typing import Any, NotRequired, TypedDict

class ResourcesConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int


class ColocationConfig(TypedDict):
    enabled: bool
    resources: NotRequired[ResourcesConfig]


class GenerationConfig(TypedDict):
    """Configuration for generation."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: NotRequired[float]
    model_name: str
    stop_token_ids: list[int] | None
    stop_strings: NotRequired[list[str] | None]
    pad_token_id: NotRequired[int]
    colocated: NotRequired[ColocationConfig]

class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by rlkit based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]
    enforce_eager: NotRequired[bool]
    extra_cli_args: NotRequired[list[str]]


class HttpVllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    gpus_per_node: int
    server_timeout: NotRequired[int]
    num_nodes: NotRequired[int]
    vllm_kwargs: NotRequired[dict[str, Any]]
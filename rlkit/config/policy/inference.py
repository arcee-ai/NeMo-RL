"""Configuration for inference."""

from typing import Any

from pydantic import BaseModel

from .resources import ResourcesConfig


class InferenceConfig(BaseModel):
    """Configuration options for Ray + vLLM Python API inference."""

    sampling_args: dict[str, Any]
    resources: ResourcesConfig

    # Parallelism
    tp_size: int = 1
    pp_size: int = 1
    enable_expert_parallel: bool = False

    # Engine settings
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 2048
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool = True
    disable_log_stats: bool = True

    # Chat template controls (HF tokenizer)
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    add_special_tokens: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] = {}

    # Tool parsing
    tool_call_parser: str | None = None

    # Startup and extra engine args
    startup_timeout_s: int = 120
    extra_engine_args: dict[str, Any] = {}

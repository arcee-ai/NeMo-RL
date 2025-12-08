"""Configuration for inference."""

from typing import Any

from pydantic import BaseModel

from .resources import ResourcesConfig


class InferenceConfig(BaseModel):
    """Configuration options for the vLLM HTTP server."""

    sampling_args: dict[str, Any]

    resources: ResourcesConfig

    tp_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"

    server_timeout: int = 60

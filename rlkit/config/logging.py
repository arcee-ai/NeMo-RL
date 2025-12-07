"""Configuration for logging."""
from typing import TypedDict, NotRequired

class WandbConfig(TypedDict):
    """Configuration for Weights & Biases logging."""
    project: NotRequired[str | None]
    name: NotRequired[str | None]

class GPUMonitoringConfig(TypedDict):
    """Configuration for GPU monitoring."""
    collection_interval: int | float
    flush_interval: int | float

class LoggerConfig(TypedDict):
    """Configuration for all logging."""
    log_dir: str
    wandb_enabled: bool
    wandb: WandbConfig
    monitor_gpus: bool
    gpu_monitoring: GPUMonitoringConfig

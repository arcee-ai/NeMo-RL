"""Configuration for logging."""

from pydantic import BaseModel


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging."""

    project: str
    name: str


class GPUMonitoringConfig(BaseModel):
    """Configuration for GPU monitoring."""

    collection_interval: int | float
    flush_interval: int | float


class LoggingConfig(BaseModel):
    """Configuration for all logging."""

    log_dir: str
    wandb: WandbConfig
    gpu_monitoring: GPUMonitoringConfig

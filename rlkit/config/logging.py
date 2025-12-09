"""Configuration for logging."""

from pydantic import BaseModel


class GPUMonitoringConfig(BaseModel):
    """Configuration for GPU monitoring."""

    collection_interval: int | float
    flush_interval: int | float


class LoggingConfig(BaseModel):
    """Configuration for all logging."""

    log_dir: str
    gpu_monitoring: GPUMonitoringConfig

from dataclasses import dataclass

@dataclass
class WandbConfig:
    project: str | None = None
    name: str | None = None

@dataclass
class TensorboardConfig:
    log_dir: str | None = None

@dataclass
class MLflowConfig:
    experiment_name: str
    run_name: str
    tracking_uri: str | None = None

@dataclass
class GPUMonitoringConfig:
    collection_interval: int | float
    flush_interval: int | float

@dataclass
class LoggerConfig:
    log_dir: str
    wandb_enabled: bool
    tensorboard_enabled: bool
    mlflow_enabled: bool
    wandb: WandbConfig
    tensorboard: TensorboardConfig
    monitor_gpus: bool
    gpu_monitoring: GPUMonitoringConfig
    num_val_samples_to_print: int
    mlflow: MLflowConfig | None = None

from typing import TypedDict, NotRequired

class WandbConfig(TypedDict):
    project: NotRequired[str | None]
    name: NotRequired[str | None]

class TensorboardConfig(TypedDict):
    log_dir: NotRequired[str | None]

class MLflowConfig(TypedDict):
    experiment_name: str
    run_name: str
    tracking_uri: NotRequired[str | None]

class GPUMonitoringConfig(TypedDict):
    collection_interval: int | float
    flush_interval: int | float

class LoggerConfig(TypedDict):
    log_dir: str
    wandb_enabled: bool
    tensorboard_enabled: bool
    mlflow_enabled: bool
    wandb: WandbConfig
    tensorboard: TensorboardConfig
    monitor_gpus: bool
    gpu_monitoring: GPUMonitoringConfig
    num_val_samples_to_print: int
    mlflow: NotRequired[MLflowConfig | None]

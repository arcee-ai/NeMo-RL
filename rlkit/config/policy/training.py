"""Configuration for training workers."""

from typing import Any

from pydantic import BaseModel, Field

from rlkit.config.policy.resources import ResourcesConfig

from .loss import LossConfig


class ParallelismConfig(BaseModel):
    """Configuration for training parallelism."""

    tp_size: int = 1
    ep_size: int = 1
    dp_replicate: int = 1


class SinglePytorchSchedulerConfig(BaseModel):
    """Configuration for a single Pytorch scheduler."""

    name: str
    kwargs: dict[str, Any]


class SchedulerConfig(BaseModel):
    """Configuration for scheduling."""

    phases: list[SinglePytorchSchedulerConfig]
    milestones: list[int] = Field(default_factory=list)


class PytorchOptimizerConfig(BaseModel):
    """Configuration for a PyTorch optimizer."""

    name: str

    # Scheduler config
    scheduler: SchedulerConfig

    # Optimizer options
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # Non-scalar optimizer options (e.g. Muon)
    scalar_optim: str | None = None
    scalar_optim_kwargs: dict[str, Any] = Field(default_factory=dict)
    pass_device_mesh: bool = False
    non_muon_params: list[str] = Field(
        default_factory=lambda: ["output.weight", "tok_embeddings.weight"]
    )


class TrainingConfig(BaseModel):
    """Configuration for training workers."""

    global_num_bins: int
    micro_batch_size: int

    parallelism: ParallelismConfig

    optimizer: PytorchOptimizerConfig

    loss: LossConfig

    resources: ResourcesConfig

    dtype: str = "bfloat16"
    activation_checkpointing: bool = True
    max_grad_norm: float | None = None

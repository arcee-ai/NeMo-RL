"""Common configuration options between inference and training."""
from typing import Any, TypedDict, NotRequired

from rlkit.config.rl.vllm import HttpVllmConfig

from .dtv2 import DTensorV2Config


class PytorchOptimizerConfig(TypedDict):
    """Configuration for a PyTorch optimizer."""
    name: str
    scalar_optim: NotRequired[str]
    scalar_optim_kwargs: NotRequired[dict[str, Any]]
    pass_device_mesh: NotRequired[bool]
    non_muon_params: NotRequired[list[str]]
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    """Configuration for a single Pytorch scheduler."""
    name: str
    kwargs: dict[str, Any]
    milestones: NotRequired[list[int]]


SchedulerMilestones = dict[str, list[int]]


class PolicyConfig(TypedDict):
    """Common configuration options between inference and training."""
    model_name: str
    train_global_batch_size: int
    train_micro_batch_size: int
    precision: str
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    logprob_batch_size: NotRequired[int]
    logprob_chunk_size: NotRequired[int]
    generation: HttpVllmConfig
    dtensor_v2_cfg: DTensorV2Config
    max_grad_norm: NotRequired[float | int]
    optimizer: PytorchOptimizerConfig
    scheduler: list[SinglePytorchSchedulerConfig]

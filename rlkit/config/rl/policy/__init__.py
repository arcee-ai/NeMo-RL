from typing import Any, Union, TypedDict, NotRequired

from rlkit.config.rl.vllm import GenerationConfig, HttpVllmConfig

from .dtv2 import DTensorV2Config

class TokenizerConfig(TypedDict):
    name: str
    chat_template: NotRequired[str | None]


class SequencePackingConfig(TypedDict):
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    algorithm: str


class PytorchOptimizerConfig(TypedDict):
    name: str
    scalar_optim: NotRequired[str]
    scalar_optim_kwargs: NotRequired[dict[str, Any]]
    pass_device_mesh: NotRequired[bool]
    non_muon_params: NotRequired[list[str]]
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]
    milestones: NotRequired[list[int]]


SchedulerMilestones = dict[str, list[int]]


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
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

from typing import Any, Union, TypedDict, NotRequired

from rlkit.config.rl.vllm import GenerationConfig

from .dtv2 import DTensorV2Config

class TokenizerConfig(TypedDict):
    name: str
    chat_template: NotRequired[str | None]


class SequencePackingConfig(TypedDict):
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    algorithm: str


class RewardModelConfig(TypedDict):
    enabled: bool
    reward_model_type: str


class PytorchOptimizerConfig(TypedDict):
    name: str
    scalar_optim: NotRequired[str | None]
    scalar_optim_kwargs: NotRequired[dict[str, Any] | None]
    pass_device_mesh: NotRequired[bool | None]
    non_muon_params: NotRequired[list[str] | None]
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]
    milestones: NotRequired[list[int] | None]


SchedulerMilestones = dict[str, list[int]]


class DynamicBatchingConfig(TypedDict):
    enabled: bool
    train_mb_tokens: NotRequired[int | None]
    logprob_mb_tokens: NotRequired[int | None]
    sequence_length_round: NotRequired[int | None]


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    precision: str
    dynamic_batching: DynamicBatchingConfig
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    logprob_batch_size: NotRequired[int | None]
    logprob_chunk_size: NotRequired[int | None]
    generation: NotRequired[GenerationConfig | None]
    generation_batch_size: NotRequired[int | None]
    reward_model_cfg: NotRequired[RewardModelConfig | None]
    dtensor_v2_cfg: NotRequired[DTensorV2Config | None]
    sequence_packing: NotRequired[SequencePackingConfig | None]
    max_grad_norm: NotRequired[Union[float, int, None]]
    refit_buffer_size_gb: NotRequired[float | None]
    optimizer: NotRequired[PytorchOptimizerConfig | None]
    scheduler: NotRequired[list[SinglePytorchSchedulerConfig] | SchedulerMilestones | None]

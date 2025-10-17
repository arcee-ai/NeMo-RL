from typing import Any, Union, TypedDict, NotRequired

from .dtv1 import DTensorConfig
from .dtv2 import DTensorV2Config

class TokenizerConfig(TypedDict):
    name: str
    chat_template: NotRequired[str | None]


class ResourcesConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int


class ColocationConfig(TypedDict):
    enabled: bool
    resources: NotRequired[ResourcesConfig | None]


class GenerationConfig(TypedDict):
    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    model_name: str
    stop_token_ids: list[int]
    stop_strings: NotRequired[list[str] | None]
    pad_token_id: NotRequired[int | None]
    colocated: NotRequired[ColocationConfig | None]


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
    use_liger_kernels: NotRequired[bool]
    logprob_batch_size: NotRequired[int | None]
    logprob_chunk_size: NotRequired[int | None]
    generation: NotRequired[GenerationConfig | None]
    generation_batch_size: NotRequired[int | None]
    reward_model_cfg: NotRequired[RewardModelConfig | None]
    dtensor_cfg: NotRequired[DTensorConfig | None]
    dtensor_v2_cfg: NotRequired[DTensorV2Config | None]
    sequence_packing: NotRequired[SequencePackingConfig | None]
    max_grad_norm: NotRequired[Union[float, int, None]]
    refit_buffer_size_gb: NotRequired[float | None]
    optimizer: NotRequired[PytorchOptimizerConfig | None]
    scheduler: NotRequired[list[SinglePytorchSchedulerConfig] | SchedulerMilestones | None]

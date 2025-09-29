from dataclasses import dataclass
from typing import Any, Union

from .dtv1 import DTensorConfig
from .dtv2 import DTensorV2Config
from .megatron import MegatronConfig

@dataclass
class TokenizerConfig:
    name: str
    chat_template: str | None = None


@dataclass
class ResourcesConfig:
    gpus_per_node: int
    num_nodes: int


@dataclass
class ColocationConfig:
    enabled: bool
    resources: ResourcesConfig | None = None


@dataclass
class GenerationConfig:
    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    model_name: str
    stop_token_ids: list[int]
    stop_strings: list[str] | None = None
    pad_token_id: int | None = None
    colocated: ColocationConfig | None = None


@dataclass
class SequencePackingConfig:
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    algorithm: str


@dataclass
class RewardModelConfig:
    enabled: bool
    reward_model_type: str


@dataclass
class PytorchOptimizerConfig:
    name: str
    kwargs: dict[str, Any]


@dataclass
class SinglePytorchSchedulerConfig:
    name: str
    kwargs: dict[str, Any]
    milestones: list[int] | None = None  # Used in SequentialLR configuration


SchedulerMilestones = dict[str, list[int]]


@dataclass
class DynamicBatchingConfig:
    # dynamic_batching improves performance by ensuring logprob and training microbatches
    # have a sufficent number of tokens to maximize GPU utilization. Specifically, variable length
    # responses are sorted by sequence length and bucketed into microbatches with a total
    # amount of tokens is approximately close to 'train_mb_tokens' and 'logprob_mb_tokens' for the
    # training and logprob stages respectively.
    enabled: bool

    # Required if enabled is true
    train_mb_tokens: int | None = None
    logprob_mb_tokens: int | None = None
    sequence_length_round: int | None = None


@dataclass
class PolicyConfig:
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    precision: str
    dynamic_batching: DynamicBatchingConfig
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    logprob_batch_size: int | None = None
    logprob_chunk_size: int | None = None
    generation: GenerationConfig | None = None
    generation_batch_size: int | None = None  # used in static batched (framework) generation
    reward_model_cfg: RewardModelConfig | None = None
    dtensor_cfg: DTensorConfig | None = None
    megatron_cfg: MegatronConfig | None = None
    dtensor_v2_cfg: DTensorV2Config | None = None
    sequence_packing: SequencePackingConfig | None = None
    max_grad_norm: Union[float, int, None] = None
    refit_buffer_size_gb: float | None = None
    optimizer: PytorchOptimizerConfig | None = None
    scheduler: list[SinglePytorchSchedulerConfig] | SchedulerMilestones | None = None

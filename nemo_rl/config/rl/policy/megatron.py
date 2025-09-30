from typing import TypedDict, NotRequired

class MegatronOptimizerConfig(TypedDict):
    optimizer: str
    lr: float
    min_lr: float
    weight_decay: float
    bf16: bool
    fp16: bool
    params_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    sgd_momentum: float
    use_distributed_optimizer: bool
    use_precision_aware_optimizer: bool
    clip_grad: float

class MegatronSchedulerConfig(TypedDict):
    start_weight_decay: float
    end_weight_decay: float
    weight_decay_incr_style: str
    lr_decay_style: str
    lr_decay_iters: int
    lr_warmup_iters: int
    lr_warmup_init: float

class MegatronDDPConfig(TypedDict):
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    average_in_collective: bool
    use_custom_fsdp: bool
    data_parallel_sharding_strategy: str

class MegatronConfig(TypedDict):
    enabled: bool
    empty_unused_memory_level: int
    activation_checkpointing: bool
    converter_type: str
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int
    num_layers_in_last_pipeline_stage: int
    context_parallel_size: int
    pipeline_dtype: str
    sequence_parallel: bool
    freeze_moe_router: bool
    expert_tensor_parallel_size: int
    expert_model_parallel_size: int
    distributed_data_parallel_config: MegatronDDPConfig
    defer_fp32_logits: NotRequired[bool | None]
    optimizer: NotRequired[MegatronOptimizerConfig | None]
    scheduler: NotRequired[MegatronSchedulerConfig | None]

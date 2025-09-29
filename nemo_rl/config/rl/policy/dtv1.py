from dataclasses import dataclass

@dataclass
class DTensorConfig:
    enabled: bool
    cpu_offload: bool | None = None
    sequence_parallel: bool | None = None
    activation_checkpointing: bool | None = None
    tensor_parallel_size: int | None = None
    context_parallel_size: int | None = None
    custom_parallel_plan: str | None = None

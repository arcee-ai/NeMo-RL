from typing import TypedDict

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.data import DataConfig
from rlkit.config.logging import LoggerConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig


class SFTConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int
    
    run_vram_torture_test: bool  # Ignore prompts and run all training on a full context window of nonsense.
    use_cut_cross_entropy: bool  # Use cut-cross-entropy loss kernel


class SFTMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

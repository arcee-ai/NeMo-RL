"""SFT-specific configuration options."""

from typing import Literal

from pydantic import BaseModel

from .checkpointing import CheckpointingConfig
from .logging import LoggingConfig
from .policy import PolicyConfig


class SFTTrainerConfig(BaseModel):
    """SFT-specific configuration options."""

    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    vram_torture_test: bool  # Ignore prompts and run all training on a full context window of nonsense.


DatasetType = Literal["axolotl", "openai_prompt_completion", "openai", "sharegpt", "native"]


class DatasetConfig(BaseModel):
    """Configuration for datasets."""

    dataset_name: str
    dataset_type: DatasetType = "native"
    hf_dataset_format: Literal["hub", "disk"] = "hub"
    shuffle: bool = False


class SFTConfig(BaseModel):
    """Root configuration for SFT runs."""

    sft: SFTTrainerConfig
    policy: PolicyConfig
    data: DatasetConfig
    logging: LoggingConfig
    checkpointing: CheckpointingConfig

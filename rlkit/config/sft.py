"""Configuration for SFT training runs."""

from typing import Literal

from pydantic import BaseModel

from rlkit.config.base import BaseConfig


class SFTTrainerConfig(BaseModel):
    """SFT-specific training configuration."""

    # Maximum number of training steps (0 = unlimited, train until data exhausted)
    max_steps: int = 0

    # Validation settings
    val_period: int = 0  # Steps between validation runs (0 = no validation)
    val_num_bins: int = 0  # Number of batches per validation (0 = full validation set)
    val_at_start: bool = False  # Run validation before training starts


DatasetType = Literal["axolotl", "openai_prompt_completion", "openai", "sharegpt", "native"]


class DataConfig(BaseModel):
    """Configuration for SFT datasets."""

    # Dataset identifier (HF hub name or local path)
    train_dataset: str
    val_dataset: str | None = None

    # Dataset splits to use
    train_split: str = "train"
    val_split: str = "validation"

    # Format of the dataset
    dataset_type: DatasetType = "native"

    # Whether to load from disk (vs HF hub)
    from_disk: bool = False

    # Whether to shuffle the training data
    shuffle: bool = True


class SFTConfig(BaseConfig):
    """Root configuration for SFT runs."""

    sft: SFTTrainerConfig
    data: DataConfig

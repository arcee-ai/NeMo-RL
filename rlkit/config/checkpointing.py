"""Configuration for checkpoint management."""

from pydantic import BaseModel


class CheckpointingConfig(BaseModel):
    """Configuration for checkpoint management."""

    enabled: bool
    checkpoint_dir: str
    save_period: int
    keep_top_k: int | None = None

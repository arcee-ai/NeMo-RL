from dataclasses import dataclass
from typing import Any, Union
import os

PathLike = Union[str, "os.PathLike[Any]"]

@dataclass
class CheckpointingConfig:
    """Configuration for checkpoint management.

    Attributes:
    enabled (bool): Whether checkpointing is enabled.
    checkpoint_dir (PathLike): Directory where checkpoints will be saved.
    metric_name (str | None): Name of the metric to use for determining best checkpoints.
    higher_is_better (bool): Whether higher values of the metric indicate better performance.
    keep_top_k (Optional[int]): Number of best checkpoints to keep. If None, all checkpoints are kept.
    """

    enabled: bool
    checkpoint_dir: PathLike
    metric_name: str | None
    higher_is_better: bool
    save_period: int
    keep_top_k: int | None = None
    checkpoint_must_save_by: str | None = None

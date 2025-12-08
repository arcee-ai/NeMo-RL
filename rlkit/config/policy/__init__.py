"""Policy configuration options common to inference and training."""

from pydantic import BaseModel

from .inference import InferenceConfig
from .training import TrainingConfig


class PolicyConfig(BaseModel):
    """Common configuration options between inference and training."""

    model_name: str

    max_total_sequence_length: int

    training: TrainingConfig
    inference: InferenceConfig | None = None

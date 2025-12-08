"""Configuration for loss functions."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ClippedPGLossConfig(BaseModel):
    """Configuration for the clipped policy gradient loss function."""

    loss_fn: Literal["clipped_pg"] = "clipped_pg"

    ratio_clip_min: float
    ratio_clip_max: float
    use_importance_sampling_correction: bool
    disable_ppo_ratio: bool = False
    ratio_clip_c: float | None = None


class CISPOLossConfig(BaseModel):
    """Configuration for the CISPO loss function."""

    loss_fn: Literal["cispo"] = "cispo"

    epsilon_max: float


class NLLLossConfig(BaseModel):
    """Configuration for the NLL loss function."""

    loss_fn: Literal["nll"] = "nll"


class CutCrossEntropyLossConfig(BaseModel):
    """Configuration for the cut cross entropy loss function."""

    loss_fn: Literal["cut_cross_entropy"] = "cut_cross_entropy"


# Discriminated union - Pydantic picks the right type based on "loss_fn" value
LossConfig = Annotated[
    ClippedPGLossConfig | CISPOLossConfig | NLLLossConfig | CutCrossEntropyLossConfig,
    Field(discriminator="loss_fn"),
]

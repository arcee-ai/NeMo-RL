"""Configuration for loss functions."""
from typing import TypedDict

class ClippedPGLossConfig(TypedDict):
    """Configuration for the clipped policy gradient loss function."""
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool

class CISPOLossConfig(TypedDict):
    """Configuration for the CISPO loss function."""
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    use_on_policy_kl_approximation: bool
    token_level_loss: bool
    epsilon_max: float
from typing import TypedDict

class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool


class CISPOLossConfig(TypedDict):
    """Configuration for CISPO (Clipped IS-weight Policy Optimization) loss.
    
    CISPO clips the importance-sampling weight and stops its gradient,
    rather than clipping the token update. This keeps gradients for all tokens
    while bounding their magnitude.
    
    References:
        - ScaleRL: https://arxiv.org/abs/2510.08475
        - MiniMax-M1: https://arxiv.org/abs/2506.09419
    """
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    use_on_policy_kl_approximation: bool
    token_level_loss: bool
from typing import TypedDict, NotRequired

class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool
    # Optional knobs (default off) to enable CISPO/ScaleRL flavor without changing defaults
    disable_ppo_ratio: NotRequired[bool]
    cispo_clip_is_weights: NotRequired[bool]
    cispo_eps_low_is: NotRequired[float]
    cispo_eps_high_is: NotRequired[float]
    cispo_use_curr_vs_gen: NotRequired[bool]
    cispo_use_unified_mask: NotRequired[bool]
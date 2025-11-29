from typing import TypedDict

class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool
    icepop_enabled: bool      # Enable IcePop masking for engine mismatch stability
    icepop_alpha: float       # Lower bound of acceptable mismatch ratio (default: 0.5)
    icepop_beta: float        # Upper bound of acceptable mismatch ratio (default: 2.0)

class CISPOLossConfig(TypedDict):
    epsilon_max: float
    reference_policy_kl_penalty: float
    use_on_policy_kl_approximation: bool
    token_level_loss: bool
    icepop_enabled: bool      # Enable IcePop masking for engine mismatch stability
    icepop_alpha: float       # Lower bound of acceptable mismatch ratio (default: 0.5)
    icepop_beta: float        # Upper bound of acceptable mismatch ratio (default: 2.0)
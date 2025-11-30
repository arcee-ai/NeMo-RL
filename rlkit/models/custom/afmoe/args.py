from dataclasses import dataclass, field
from typing_extensions import Literal

from torch import nn

from rlkit.models.custom.model import BaseModelArgs

@dataclass
class MoEArgs:
    num_experts: int = 16
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0
    score_before_experts: bool = False

    # token-choice
    top_k: int = 2
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3
    
    _debug_force_load_balance: bool = False

@dataclass
class AFMoEModelArgs(BaseModelArgs):
    dim: int = 4096
    inter_dim: int = 16384
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int = 128256
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    global_attn_every_n_layers: int = 4

    moe_inter_dim: int = 512
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    n_dense_layers: int = 3

    max_seq_len: int = 131072
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    local_attn_mask_type: str = "causal_sliding_window"
    local_attn_sliding_window_size: int = 1024
    use_sdpa_for_global_attn: bool = False
    eos_id: int = 0

    mup_enabled: bool = False

    enable_weight_tying: bool = False
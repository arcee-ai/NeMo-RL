from dataclasses import dataclass, field

from rlkit.models.custom.model import BaseModelArgs
from rlkit.models.custom.moe import MoEArgs

@dataclass
class Qwen3MoEModelArgs(BaseModelArgs):
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072
    
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)
    moe_intermediate_size: int = 768

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    rope_scaling: dict | None = None
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    fixed_block_size: int | None = None
    eos_id: int = 151645

    enable_weight_tying: bool = False
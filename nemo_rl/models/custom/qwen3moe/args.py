from dataclasses import dataclass

from nemo_rl.models.custom.model import BaseModelArgs
from nemo_rl.models.custom.moe import MoEArgs

@dataclass
class Qwen3MoEModelArgs(BaseModelArgs):
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072
    
    moe_args: MoEArgs = MoEArgs()
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = []

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 151645

    enable_weight_tying: bool = False
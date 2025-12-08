"""Convert HuggingFace configs to native model classes and arguments."""
from transformers import PretrainedConfig

from rlkit.models import BaseModel, BaseModelArgs

from .afmoe.args import AFMoEModelArgs
from .afmoe.args import MoEArgs as MoEArgsAFMoE
from .afmoe.model import AFMoEModel
from .afmoe.state_dict_adapter import AFMoEStateDictAdapter
from .qwen3.args import Qwen3ModelArgs
from .qwen3.model import Qwen3Model
from .qwen3.state_dict_adapter import Qwen3StateDictAdapter
from .state_dict_adapter import BaseStateDictAdapter


def get_model_config(config: PretrainedConfig) -> tuple[type[BaseModel], BaseModelArgs, type[BaseStateDictAdapter]]:
    """Get the model class, native model arguments, and state dict adapter class for a given HuggingFace config."""
    mt = config.model_type

    if mt == "afmoe":
        layer_types = config.layer_types
        glob_attn_every_n = layer_types.index("full_attention") + 1

        return AFMoEModel, AFMoEModelArgs(
            dim = config.hidden_size,
            inter_dim = config.intermediate_size,
            n_layers = config.num_hidden_layers,
            n_heads = config.num_attention_heads,
            n_kv_heads = config.num_key_value_heads,
            head_dim = config.head_dim,
            vocab_size = config.vocab_size,
            norm_eps = config.rms_norm_eps,
            rope_theta = config.rope_theta,
            global_attn_every_n_layers = glob_attn_every_n,
            moe_inter_dim = config.moe_intermediate_size,
            moe_args = MoEArgsAFMoE(
                num_experts = config.num_experts,
                num_shared_experts = config.num_shared_experts,
                score_func = config.score_func,
                route_norm = config.route_norm,
                route_scale = config.route_scale,
                score_before_experts = False,
                top_k = config.num_experts_per_tok,
                use_grouped_mm = True,
                load_balance_coeff = config.load_balance_coeff,
            ),
            n_dense_layers = config.num_dense_layers,
            max_seq_len = config.max_position_embeddings,
            depth_init = False,
            use_flex_attn=True,
            attn_mask_type="block_causal",
            local_attn_mask_type="causal_sliding_window",
            local_attn_sliding_window_size=config.sliding_window,
            mup_enabled = config.mup_enabled,
            enable_weight_tying = config.tie_word_embeddings,
        ), AFMoEStateDictAdapter
    elif mt == "qwen3":
        uses_sliding_causal = getattr(config, "sliding_window", None) is not None
        return Qwen3Model, Qwen3ModelArgs(
            dim = config.hidden_size,
            n_layers = config.num_hidden_layers,
            n_heads = config.num_attention_heads,
            n_kv_heads = config.num_key_value_heads,
            vocab_size = config.vocab_size,
            head_dim = config.head_dim,
            hidden_dim = config.intermediate_size,
            norm_eps = config.rms_norm_eps,
            rope_theta = config.rope_theta,
            max_seq_len = config.max_position_embeddings,
            eos_id = int(config.eos_token_id) if config.eos_token_id is not None else 0,
            enable_weight_tying = config.tie_word_embeddings,
            attn_mask_type = "block_causal",
            use_flex_attn = uses_sliding_causal,
            fixed_block_size = config.sliding_window if uses_sliding_causal else None,
        ), Qwen3StateDictAdapter
    else:
        raise ValueError(f"Model type {mt} unknown or not supported")

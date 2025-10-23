from typing import Callable
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from rlkit.models.custom.model import BaseModelArgs

from rlkit.models.custom.afmoe.model import AFMoEModel
from rlkit.models.custom.afmoe.args import AFMoEModelArgs, MoEArgs as MoEArgsAFMoE
from rlkit.models.custom.afmoe.state_dict_adapter import AFMoEStateDictAdapter
from rlkit.models.custom.afmoe.parallelize import parallelize_afmoe

from rlkit.models.custom.moe import MoEArgs
from rlkit.models.custom.qwen3.model import Qwen3Model
from rlkit.models.custom.qwen3.args import Qwen3ModelArgs
from rlkit.models.custom.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from rlkit.models.custom.qwen3.parallelize import parallelize_qwen3

from rlkit.models.custom.qwen3moe.model import Qwen3MoEModel
from rlkit.models.custom.qwen3moe.args import Qwen3MoEModelArgs
from rlkit.models.custom.qwen3moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from rlkit.models.custom.qwen3moe.parallelize import parallelize_qwen3moe

from rlkit.models.custom.state_dict_adapter import BaseStateDictAdapter

def get_model_config(config: PretrainedConfig) -> tuple[type[nn.Module], BaseModelArgs, type[BaseStateDictAdapter], Callable]:
    mt = config.model_type
    
    is_nightly_torch = hasattr(torch, "_grouped_mm")
    
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
                use_grouped_mm = is_nightly_torch,
                load_balance_coeff = config.load_balance_coeff,
            ),
            n_dense_layers = config.num_dense_layers,
            max_seq_len = config.max_position_embeddings,
            depth_init = False,
            use_flex_attn=True,
            attn_mask_type="causal",
            local_attn_mask_type="causal_sliding_window",
            local_attn_sliding_window_size=config.sliding_window,
            mup_enabled = config.mup_enabled,
            enable_weight_tying = config.tie_word_embeddings,
        ), AFMoEStateDictAdapter, parallelize_afmoe
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
            eos_id = int(config.eos_token_id) if getattr(config, "eos_token_id", None) is not None else 0,
            enable_weight_tying = config.tie_word_embeddings,
            attn_mask_type = "sliding_causal" if uses_sliding_causal else "causal",
            use_flex_attn = uses_sliding_causal,
            fixed_block_size = config.sliding_window if uses_sliding_causal else None,
        ), Qwen3StateDictAdapter, parallelize_qwen3
    elif mt == "qwen3_moe":
        uses_sliding_causal = getattr(config, "sliding_window", None) is not None
        return Qwen3MoEModel, Qwen3MoEModelArgs(
            dim = config.hidden_size,
            n_layers = config.num_hidden_layers,
            n_heads = config.num_attention_heads,
            n_kv_heads = config.num_key_value_heads,
            vocab_size = config.vocab_size,
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            hidden_dim = config.intermediate_size,
            norm_eps = config.rms_norm_eps,
            rope_theta = config.rope_theta,
            rope_scaling = getattr(config, "rope_scaling", None),
            max_seq_len = config.max_position_embeddings,
            eos_id = int(config.eos_token_id) if config.eos_token_id is not None else 0,
            enable_weight_tying = config.tie_word_embeddings,
            attn_mask_type = "sliding_causal" if uses_sliding_causal else "causal",
            use_flex_attn = uses_sliding_causal,
            fixed_block_size = config.sliding_window if uses_sliding_causal else None,
            moe_args = MoEArgs(
                num_experts = config.num_experts,
                num_shared_experts = 0,
                score_func = "softmax",
                route_norm = config.norm_topk_prob,
                route_scale = 1,
                score_before_experts = False,
                top_k = config.num_experts_per_tok,
                use_grouped_mm = is_nightly_torch,
                load_balance_coeff = None
            ),
            decoder_sparse_step = getattr(config, "decoder_sparse_step", 1),
            mlp_only_layers = list(getattr(config, "mlp_only_layers", [])),
            moe_intermediate_size = config.moe_intermediate_size,
        ), Qwen3MoeStateDictAdapter, parallelize_qwen3moe
    else:
        raise ValueError(f"Model type {mt} unknown or not supported")
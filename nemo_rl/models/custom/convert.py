from typing import Callable
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from nemo_rl.models.custom.model import BaseModelArgs

from nemo_rl.models.custom.llama3.model import Transformer as LlamaModel
from nemo_rl.models.custom.llama3.args import TransformerModelArgs as LlamaModelArgs
from nemo_rl.models.custom.llama3.state_dict_adapter import Llama3StateDictAdapter
from nemo_rl.models.custom.llama3.parallelize import parallelize_llama

from nemo_rl.models.custom.moe import MoEArgs
from nemo_rl.models.custom.qwen3.model import Qwen3Model
from nemo_rl.models.custom.qwen3.args import Qwen3ModelArgs
from nemo_rl.models.custom.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from nemo_rl.models.custom.qwen3.parallelize import parallelize_qwen3

from nemo_rl.models.custom.qwen3moe.model import Qwen3MoEModel
from nemo_rl.models.custom.qwen3moe.args import Qwen3MoEModelArgs
from nemo_rl.models.custom.qwen3moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_rl.models.custom.qwen3moe.parallelize import parallelize_qwen3moe

from nemo_rl.models.custom.deepseekv3.model import DeepSeekV3Model
from nemo_rl.models.custom.deepseekv3.args import DeepSeekV3ModelArgs
from nemo_rl.models.custom.deepseekv3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_rl.models.custom.deepseekv3.parallelize import parallelize_dsv3

from nemo_rl.models.custom.state_dict_adapter import BaseStateDictAdapter

def get_model_config(config: PretrainedConfig) -> tuple[type[nn.Module], BaseModelArgs, type[BaseStateDictAdapter], Callable]:
    mt = config.model_type
    
    if mt == "llama":
        # TODO: verfy this works on all llama3 models
        return LlamaModel, LlamaModelArgs(
            dim = config.hidden_size,
            n_layers = config.num_hidden_layers,
            n_heads = config.num_attention_heads,
            n_kv_heads = config.num_key_value_heads,
            vocab_size = config.vocab_size,
            norm_eps = config.layer_norm_eps,
            rope_theta = config.rope_theta,
            max_seq_len = config.max_position_embeddings,
            eos_id = int(config.eos_token_id) if getattr(config, "eos_token_id", None) is not None else 0
        ), Llama3StateDictAdapter, parallelize_llama
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
                use_grouped_mm = False,
                load_balance_coeff = None
            ),
            decoder_sparse_step = getattr(config, "decoder_sparse_step", 1),
            mlp_only_layers = list(getattr(config, "mlp_only_layers", [])),
            moe_intermediate_size = config.moe_intermediate_size,
        ), Qwen3MoeStateDictAdapter, parallelize_qwen3moe
    elif mt == "deepseek_v3":
        return DeepSeekV3Model, DeepSeekV3ModelArgs(
            dim = config.hidden_size,
            inter_dim = getattr(config, "intermediate_size", 0),
            moe_inter_dim = getattr(config, "moe_intermediate_size", 0),
            n_layers = config.num_hidden_layers,
            n_dense_layers = getattr(config, "first_k_dense_replace", 1),
            n_heads = config.num_attention_heads,
            vocab_size = config.vocab_size,
            norm_eps = getattr(config, "rms_norm_eps", getattr(config, "layer_norm_eps", 1e-5)),
            rope_theta = getattr(config, "rope_theta", 10000.0),
            max_seq_len = getattr(config, "max_position_embeddings", 4096),
            q_lora_rank = 0 if getattr(config, "q_lora_rank", None) in (None, 0) else int(config.q_lora_rank),
            kv_lora_rank = getattr(config, "kv_lora_rank", 512),
            qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128),
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64),
            v_head_dim = getattr(config, "v_head_dim", 128),
            use_flex_attn = False,
            attn_mask_type = "causal",
            moe_args = MoEArgs(
                num_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0)),
                num_shared_experts = getattr(config, "n_shared_experts", 0),
                score_func = "sigmoid" if getattr(config, "scoring_func", "sigmoid") == "sigmoid" else "softmax",
                route_norm = getattr(config, "norm_topk_prob", False),
                route_scale = getattr(config, "routed_scaling_factor", 1.0),
                score_before_experts = True,
                top_k = getattr(config, "num_experts_per_tok", 1),
                use_grouped_mm = False,
                load_balance_coeff = None,
            ),
        ), DeepSeekV3StateDictAdapter, parallelize_dsv3
    else:
        raise ValueError(f"Model type {mt} unknown or not supported")
from typing import Callable
from torch import nn
from transformers import PreTrainedModel

from nemo_rl.models.custom.model import BaseModelArgs

from nemo_rl.models.custom.llama3.model import Transformer as LlamaModel
from nemo_rl.models.custom.llama3.args import TransformerModelArgs as LlamaModelArgs
from nemo_rl.models.custom.llama3.state_dict_adapter import Llama3StateDictAdapter
from nemo_rl.models.custom.llama3.parallelize import parallelize_llama

from nemo_rl.models.custom.qwen3.model import Qwen3Model
from nemo_rl.models.custom.qwen3.args import Qwen3ModelArgs
from nemo_rl.models.custom.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from nemo_rl.models.custom.qwen3.parallelize import parallelize_qwen3

from nemo_rl.models.custom.state_dict_adapter import BaseStateDictAdapter

def get_model_config(model: PreTrainedModel) -> tuple[type[nn.Module], BaseModelArgs, type[BaseStateDictAdapter], Callable]:
    mt = model.config.model_type
    
    if mt == "llama":
        # TODO: verfy this works on all llama3 models
        return LlamaModel, LlamaModelArgs(
            dim = model.config.hidden_size,
            n_layers = model.config.num_hidden_layers,
            n_heads = model.config.num_attention_heads,
            n_kv_heads = model.config.num_key_value_heads,
            vocab_size = model.config.vocab_size,
            norm_eps = model.config.layer_norm_eps,
            rope_theta = model.config.rope_theta,
            max_seq_len = model.config.max_position_embeddings,
            eos_id = model.config.eos_token_id
        ), Llama3StateDictAdapter, parallelize_llama
    elif mt == "qwen3":
        return Qwen3Model, Qwen3ModelArgs(
            dim = model.config.hidden_size,
            n_layers = model.config.num_hidden_layers,
            n_heads = model.config.num_attention_heads,
            n_kv_heads = model.config.num_key_value_heads,
            vocab_size = model.config.vocab_size,
            head_dim = model.config.head_dim,
            hidden_dim = model.config.intermediate_size,
            norm_eps = model.config.rms_norm_eps,
            rope_theta = model.config.rope_theta,
            max_seq_len = model.config.max_position_embeddings,
            eos_id = model.config.eos_token_id,
            enable_weight_tying = model.config.tie_word_embeddings
        ), Qwen3StateDictAdapter, parallelize_qwen3
    else:
        raise ValueError(f"Model type {mt} unknown or not supported")
from sympy.logic import true
import torch
import torch.nn as nn
from typing import Optional

from nemo_rl.models.custom.moe import MoE, FeedForward
from nemo_rl.models.custom.qwen3moe.args import Qwen3MoEModelArgs

from nemo_rl.models.custom.qwen3.model import Attention, FeedForward
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from nemo_rl.models.custom.attention import init_attention_mask
from nemo_rl.models.custom.model import BaseModel

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: Qwen3MoEModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        
        self.moe_enabled = (
            (layer_id not in model_args.mlp_only_layers)
            and (model_args.moe_args.num_experts > 0)
            and ((layer_id + 1) % model_args.decoder_sparse_step == 0)
        )
        
        # TODO: make moe config a superclass of non-moe config?
        self.attention = Attention(model_args) # type: ignore
        
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        if self.moe_enabled:
            # For MoE layers, experts operate on model hidden size (dim) and expand to moe_intermediate_size
            self.moe = MoE(model_args.moe_args, model_args.dim, model_args.moe_intermediate_size)
        else:
            self.ffn = FeedForward(model_args.dim, model_args.hidden_dim)
        
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5
            
    def forward(self, hidden_states: torch.Tensor, rope_cache: torch.Tensor):
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states = self.attention(hidden_states, rope_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        if self.moe_enabled:
            hidden_states = self.moe(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    def init_weights(self, buffer_device: torch.device | None = None):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

class Qwen3MoEModel(BaseModel):
    def __init__(self, model_args: Qwen3MoEModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim
        
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.model_args.dim)
        
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )
        
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(self.model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, self.model_args)
        
        self.norm = nn.RMSNorm(self.model_args.dim, eps=self.model_args.norm_eps)
        
        self.output = nn.Linear(self.model_args.dim, self.vocab_size, bias=False)
        
        self.init_weights()
    
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std * cutoff_factor,
                a=-2 * cutoff_factor,
                b=2 * cutoff_factor,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        # Build a minimal config object for HF RoPE init functions
        class _Cfg:
            rope_theta: float
            head_dim: int
            hidden_size: int
            num_attention_heads: int
            max_position_embeddings: int
            rope_scaling: dict

        cfg = _Cfg()
        cfg.rope_theta = self.model_args.rope_theta
        cfg.head_dim = self.model_args.head_dim
        cfg.hidden_size = self.model_args.dim
        cfg.num_attention_heads = self.model_args.n_heads
        cfg.max_position_embeddings = self.model_args.max_seq_len
        cfg.rope_scaling = (
            self.model_args.rope_scaling
            if self.model_args.rope_scaling is not None
            else {"rope_type": "default"}
        )

        rope_type = cfg.rope_scaling.get("rope_type", cfg.rope_scaling.get("type", "default"))
        init_fn = ROPE_INIT_FUNCTIONS.get(rope_type, ROPE_INIT_FUNCTIONS["default"])

        device = torch.device("cpu")
        inv_freq, attention_scaling = init_fn(cfg, device, seq_len=self.model_args.max_seq_len)
        # Compute cos/sin with attention scaling and concatenate to match apply_rotary_emb expectations
        seq_len = self.model_args.max_seq_len
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(position_ids, inv_freq.float())  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        rope_cache = torch.cat([cos, sin], dim=-1)  # (seq_len, head_dim)
        return rope_cache

    def forward(self, tokens: torch.Tensor):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        
        if self.model_args.use_flex_attn:
            # FlexAttention only needs seq_len; pass a dummy 4D tensor with the correct [B, S]
            dummy = torch.empty(h.shape[0], h.shape[1], 1, 1, device=h.device)
            init_attention_mask(dummy, eos_id=None)
        
        for layer in self.layers.values():
            h = layer(h, self.rope_cache)
            
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

# TODO: router aux loss
import torch
import torch.nn as nn

from nemo_rl.models.custom.moe import MoE
from nemo_rl.models.custom.qwen3moe.args import Qwen3MoEModelArgs

from nemo_rl.models.custom.qwen3.model import Attention, FeedForward, precompute_rope_cache

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: Qwen3MoEModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        
        # TODO: how does qwen3 actually decide this
        self.moe_enabled = (layer_id >= layer_id % model_args.decoder_sparse_step == 0) and \
            (layer_id not in model_args.mlp_only_layers) and \
            (model_args.moe_args.num_experts > 0)
        
        # TODO: make moe config a superclass of non-moe config?
        self.attention = Attention(model_args) # type: ignore
        
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        if self.moe_enabled:
            self.moe = MoE(model_args.moe_args, model_args.dim, model_args.hidden_dim)
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )
        
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5
            
    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), rope_cache)
        if self.moe_enabled:
            h = h + self.moe(self.ffn_norm(h))
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return h
    
    def init_weights(self, buffer_device: torch.device | None = None):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

class Qwen3MoEModel(nn.Module):
    def __init__(self, args: Qwen3MoEModelArgs):
        super().__init__()
        self.model_args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.eos_id = args.eos_id
        self.head_dim = args.head_dim
        
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
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.tok_embeddings(input_ids) if self.tok_embeddings is not None else input_ids
        
        for layer in self.layers.values():
            h = layer(h, self.rope_cache)
            
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

# TODO: router aux loss
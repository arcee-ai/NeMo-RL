from sympy.logic import true
import torch
import torch.nn as nn
from typing import Optional

from nemo_rl.models.custom.moe import MoE
from nemo_rl.models.custom.qwen3moe.args import Qwen3MoEModelArgs

from nemo_rl.models.custom.qwen3.model import Attention, FeedForward
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from nemo_rl.models.custom.attention import init_attention_mask

from transformers.models.qwen3_moe.modeling_qwen3_moe import create_causal_mask

class HFRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen3MoeMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_size = dim
        self.intermediate_size = hidden_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, num_experts: int, num_experts_per_tok: int, norm_topk_prob: bool, dim: int, hidden_dim: int, moe_hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # gating
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(dim, moe_hidden_dim) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

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
        
        self.attention_norm = HFRMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = HFRMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        if self.moe_enabled:
            # For MoE layers, experts operate on model hidden size (dim) and expand to moe_intermediate_size
            #self.moe = MoE(model_args.moe_args, model_args.dim, model_args.moe_intermediate_size)
            self.mlp = Qwen3MoeSparseMoeBlock(
                model_args.moe_args.num_experts,
                model_args.moe_args.top_k,
                True,
                model_args.dim,
                model_args.hidden_dim,
                model_args.moe_intermediate_size
            )
        else:
            self.mlp = Qwen3MoeMLP(model_args.dim, model_args.hidden_dim)
        
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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    def init_weights(self, buffer_device: torch.device | None = None):
        # for norm in (self.attention_norm, self.ffn_norm):
        #     norm.reset_parameters()
        # self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            pass
            #self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            pass
            # self.feed_forward.init_weights(self.weight_init_std)

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

    def forward(self, input_ids: torch.Tensor):
        h = self.tok_embeddings(input_ids) if self.tok_embeddings is not None else input_ids
        
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
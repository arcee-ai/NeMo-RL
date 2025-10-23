import torch
import torch.nn.functional as F
from torch import nn

from rlkit.models.custom.attention import build_attention, init_attention_mask
from rlkit.models.custom.model import BaseModel

from .args import AFMoEModelArgs
from .moe import FeedForward, MoE


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine frequencies for rotary embeddings.

    This function calculates frequency tensors and returns their cosine and sine values
    for use in rotary position embeddings.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Precomputed cosine and sine frequency tensors.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Duplicate frequencies to match full head dimension (needed for rotate_half)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using cosine and sine frequencies.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    cosine and sine frequency tensors. The rotary embeddings are applied using the rotate_half approach.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. Shape: (bs, seqlen, n_heads, head_dim)
        xk (torch.Tensor): Key tensor to apply rotary embeddings. Shape: (bs, seqlen, n_kv_heads, head_dim)
        cos (torch.Tensor): Precomputed cosine frequencies. Shape: (seqlen, head_dim)
        sin (torch.Tensor): Precomputed sine frequencies. Shape: (seqlen, head_dim)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # Get sequence length from query tensor
    seqlen = xq.shape[1]
    
    # Slice cos/sin to match sequence length and reshape for broadcasting
    # cos/sin: (seqlen, head_dim) -> (1, seqlen, 1, head_dim)
    cos = cos[:seqlen].unsqueeze(0).unsqueeze(2)
    sin = sin[:seqlen].unsqueeze(0).unsqueeze(2)
    
    # Apply rotary embeddings: x_embed = (x * cos) + (rotate_half(x) * sin)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (AFMoEModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: AFMoEModelArgs, layer_id: int):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        if model_args.head_dim is not None:
            self.head_dim = model_args.head_dim
        else:
            self.head_dim = model_args.dim // model_args.n_heads
        self.is_local_attention = (layer_id+1) % model_args.global_attn_every_n_layers != 0
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.gate_proj = nn.Linear(model_args.dim, self.head_dim * self.n_heads, bias=False)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)
        if self.is_local_attention:
            self.sdpa = build_attention(model_args.use_flex_attn, model_args.local_attn_mask_type)
        else:
            self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, out_init_std: float, base_init_std: float):
        cutoff_factor = 3
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=base_init_std,
                a=-cutoff_factor*base_init_std,
                b=cutoff_factor*base_init_std,
            ) # pyright: ignore[reportUnusedCallResult]
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=out_init_std,
            a=-cutoff_factor*out_init_std,
            b=cutoff_factor*out_init_std,
        ) # pyright: ignore[reportUnusedCallResult]

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (tuple[torch.Tensor, torch.Tensor]): Precomputed cosine and sine frequency tensors.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        gate = self.gate_proj(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)

        if self.is_local_attention:
            cos, sin = freqs_cis
            xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        output = output * F.sigmoid(gate)
        return self.wo(output)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (AFMoEModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(
        self,
        layer_id: int,
        model_args: AFMoEModelArgs,
    ):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.moe_enabled = layer_id >= model_args.n_dense_layers
        self.n_layers = model_args.n_layers

        self.attention = Attention(model_args, layer_id)
        if self.moe_enabled:
            self.moe = MoE(model_args)
        else:
            self.feed_forward = FeedForward(dim=model_args.dim, hidden_dim=model_args.inter_dim)
        self.attention_norm_a = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.attention_norm_b = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm_a = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm_b = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.base_std = 0.5*(model_args.dim**-0.5) if model_args.mup_enabled else 0.02
        if model_args.depth_init:
            self.weight_init_std = self.base_std / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = self.base_std # sandwich norm, residual-facing layers are normalized anyway

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (tuple[torch.Tensor, torch.Tensor]): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention_norm_b(self.attention(self.attention_norm_a(x), freqs_cis))
        if self.moe_enabled:
            out = h + self.ffn_norm_b(self.moe(self.ffn_norm_a(h)))
        else:
            out = h + self.ffn_norm_b(self.feed_forward(self.ffn_norm_a(h)))
        return out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm_a, self.ffn_norm_a):
            norm.reset_parameters()
        for post_norm in (self.attention_norm_b, self.ffn_norm_b):
            # paper ref: https://arxiv.org/abs/2504.07866
            nn.init.constant_(post_norm.weight, self.n_layers**-0.5)  # pyright: ignore[reportUnusedCallResult, reportAny]
        self.attention.init_weights(self.weight_init_std, self.base_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, self.base_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std, self.base_std)


class AFMoEModel(BaseModel):
    """
    Transformer Module

    Args:
        model_args (AFMoEModelArgs): Model configuration arguments.

    Attributes:
        model_args (AFMoEModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: AFMoEModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Register separate buffers for cos and sin
        freqs_cos, freqs_sin = self._precompute_freqs_cis()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cos.device
        cutoff_factor = 3
        if self.model_args.mup_enabled:
            emb_std = 0.5 * (self.model_args.dim**-0.5)
            lmh_std = 0.5 * (self.model_args.dim**-0.5)
        else: # default path
            emb_std = 1
            lmh_std = self.model_args.dim**-0.5
        with torch.device(buffer_device):
            freqs_cos, freqs_sin = self._precompute_freqs_cis()
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
        if self.tok_embeddings is not None: # pyright: ignore[reportUnnecessaryComparison]
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                std=emb_std,
                a=-cutoff_factor*emb_std,
                b=cutoff_factor*emb_std
            ) # pyright: ignore[reportUnusedCallResult]
        for layer in self.layers.values():
            if layer is not None: # pyright: ignore[reportUnnecessaryComparison]
                layer.init_weights(buffer_device=buffer_device) # pyright: ignore[reportCallIssue]
        if self.norm is not None: # pyright: ignore[reportUnnecessaryComparison]
            self.norm.reset_parameters()
        if self.output is not None: # pyright: ignore[reportUnnecessaryComparison]
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=lmh_std,
                a=-cutoff_factor*lmh_std,
                b=cutoff_factor*lmh_std,
            ) # pyright: ignore[reportUnusedCallResult]

    def _precompute_freqs_cis(self) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.model_args.head_dim if self.model_args.head_dim is not None else self.model_args.dim // self.model_args.n_heads
        return precompute_freqs_cis(
            head_dim,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        if self.model_args.use_flex_attn:
            init_attention_mask(
                input_batch if input_batch is not None else tokens,
                eos_id=eos_id,
                sliding_window_size=self.model_args.local_attn_sliding_window_size
            )

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.model_args.mup_enabled:
            h = h * (self.model_args.dim**0.5)

        freqs_cis = (self.freqs_cos, self.freqs_sin)
        for layer in self.layers.values():
            h = layer(h, freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
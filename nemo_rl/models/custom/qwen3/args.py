# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class TransformerModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = 32
    vocab_size: int = 151936
    # Qwen3 uses RMSNorm with eps 1e-6
    norm_eps: float = 1e-6
    rope_theta: float = 10000

    # Explicit head dimension (Qwen3 default is 128)
    head_dim: int = 128
    # Explicit intermediate size (Qwen3-8B uses 22016)
    intermediate_size: int = 22016

    # Sequence length and initialization
    max_seq_len: int = 32768
    depth_init: bool = True

    # Attention controls
    use_flex_attn: bool = True
    attn_mask_type: str = "causal"
    attention_dropout: float = 0.0

    # Sliding window attention controls
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28

    # Misc
    eos_id: int = 0



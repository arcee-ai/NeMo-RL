# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

from dataclasses import dataclass

from nemo_rl.models.custom.model import BaseModelArgs

@dataclass
class Qwen3ModelArgs(BaseModelArgs):

    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 151645

    enable_weight_tying: bool = False



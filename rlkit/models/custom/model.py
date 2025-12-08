# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.
from rlkit.models.custom.attention import AttentionMasksType

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from torch import nn
import torch

from transformers import PreTrainedModel

@dataclass
class BaseModelArgs:
    """All ModelArgs should inherit from this class.

    The only usage of this class is type checking but allows us to extend common
    arguments to all models in the future.
    """

    _enforced: str = "This field is used to enforce all fields have defaults."

class BaseModel(nn.Module):
    @abstractmethod
    def __init__(self, model_args: BaseModelArgs, skip_logits: bool = False):
        super().__init__()
    
    @abstractmethod
    def forward(self, tokens: torch.Tensor):
        pass
    
    layers: nn.ModuleDict
    
    # LM head, used in cut cross-entropy
    output: nn.Linear
    
    @abstractmethod
    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        separator_value: int,
    ) -> AttentionMasksType | None:
        pass
    
    def collect_router_statistics(
        self, ep_mesh=None, as_fractions: bool = False
    ) -> dict[str, float]:
        """Collect router statistics from all MoE layers. Default implementation returns an empty dict."""
        return {}
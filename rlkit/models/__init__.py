"""RLKit models package with base classes."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.
from abc import ABC
from .attention import AttentionMasksType

from abc import abstractmethod
from dataclasses import dataclass

from torch import nn
import torch

@dataclass
class BaseModelArgs(ABC):
    """Abstract base class for all model arguments objects."""

    _enforced: str = "This field is used to enforce all fields have defaults."

class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    @abstractmethod
    def __init__(self, model_args: BaseModelArgs, skip_logits: bool = False):
        """Initialize the model."""
        super().__init__()
    
    @abstractmethod
    def forward(self, tokens: torch.Tensor):
        """Forward pass for the model."""
        pass
    
    layers: nn.ModuleDict
    
    # LM head, used in cut cross-entropy
    output: nn.Linear
    
    @abstractmethod
    def get_attention_masks(
        self,
        input_ids: torch.Tensor,
        separator_value: int,
    ) -> AttentionMasksType | None:
        """Generate attention masks for a given sequence.
        
        Args:
            input_ids (torch.Tensor): The token IDs of the sequence.
            separator_value (int): The token ID separating packed documents.
        
        Returns:
            AttentionMasksType | None: Attention masks, or None if the configuration does not support attention masking.
        """
        pass
    
    def collect_router_statistics(
        self, ep_mesh=None, as_fractions: bool = False
    ) -> dict[str, float]:
        """Collect routing statistics from all MoE layers.
        
        MoE models should override this method to collect routing statistics from all MoE layers.
        
        Args:
            ep_mesh (DeviceMesh | None): The device mesh for expert parallelism.
            as_fractions (bool): Whether to return the statistics as fractions.
        
        Returns:
            dict[str, float]: Routing statistics accumulated by the model, or an empty dict if the model does not support routing statistics.
        """
        return {}
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

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

    @abstractmethod
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        pass

class BaseModel(nn.Module):
    @abstractmethod
    def __init__(self, model_args: BaseModelArgs):
        super().__init__()
    
    @abstractmethod
    def forward(self, tokens: torch.Tensor):
        pass
"""Abstract base class for state dict transformations."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger()

from . import BaseModelArgs


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    
    Args:
        model_args: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    @abstractmethod
    def __init__(self, model_args: BaseModelArgs, hf_assets_path: str | None):
        """Initialize the state dict adapter."""
        pass

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from HuggingFace state dict to native model format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        pass

    def get_hf_metadata(self, state_dict: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Return HF metadata as {name: (shape, dtype)}.

        Default implementation materializes the full converted state dict.
        Adapters with expensive conversions should override this.
        """
        hf_state_dict = self.to_hf(state_dict)
        metadata = {}
        for name, tensor in hf_state_dict.items():
            metadata[name] = (tensor.shape, tensor.dtype)
        return metadata

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, NotRequired, TypedDict

import torch


class ResourcesConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int


class ColocationConfig(TypedDict):
    enabled: bool
    resources: NotRequired[ResourcesConfig]


class GenerationConfig(TypedDict):
    """Configuration for generation."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: NotRequired[float]
    model_name: str
    stop_token_ids: list[int] | None
    stop_strings: NotRequired[list[str] | None]
    pad_token_id: NotRequired[int]
    colocated: NotRequired[ColocationConfig]


class GenerationDatumSpec(TypedDict):
    """Specification for input data required by generation models.

    - input_ids: Tensor of token IDs representing the input sequences (right padded)
    - input_lengths: Tensor containing the actual length of each sequence (without padding)
    - stop_strings: Optional list of strings to stop generation (per sample)
    - __extra__: Additional model-specific data fields

    Example of a batch with 4 entries with different sequence lengths:
    ```
    # Batch of 4 sequences with lengths [3, 5, 2, 4]

    input_ids (padded):
    [
      [101, 2054, 2003,    0,    0],  # Length 3
      [101, 2054, 2003, 2001, 1996],  # Length 5
      [101, 2054,    0,    0,    0],  # Length 2
      [101, 2054, 2003, 2001,    0],  # Length 4
    ]

    input_lengths:
    [3, 5, 2, 4]
    ```

    All functions receiving or returning GenerationDatumSpec should ensure
    right padding is maintained.
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    stop_strings: NotRequired[list[str]]
    __extra__: Any


class GenerationOutputSpec(TypedDict):
    """Specification for output data returned by generation models.

    - output_ids: Tensor of token IDs representing the generated sequences (right padded)
    - generation_lengths: Tensor containing the actual length of each generated sequence
    - unpadded_sequence_lengths: Tensor containing the actual length of each input + generated sequence (without padding)
    - logprobs: Tensor of log probabilities for each generated token (right padded with zeros)
    - __extra__: Additional model-specific data fields

    Example of a batch with 2 sequences:
    ```
    # Sample batch with 2 examples
    # - Example 1: Input length 3, generated response length 4
    # - Example 2: Input length 5, generated response length 2

    output_ids (right-padded):
    [
      [101, 2054, 2003, 2023, 2003, 1037, 2200,    0],  # 7 valid tokens (3 input + 4 output)
      [101, 2054, 2003, 2001, 1996, 3014, 2005,    0],  # 7 valid tokens (5 input + 2 output)
    ]

    generation_lengths:
    [4, 2]  # Length of just the generated response part

    unpadded_sequence_lengths:
    [7, 7]  # Length of full valid sequence (input + generated response)

    logprobs (right-padded with zeros):
    [
      [0.0, 0.0, 0.0, -1.2, -0.8, -2.1, -1.5, 0.0],  # First 3 are 0 (input tokens), next 4 are actual logprobs
      [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, -1.7, 0.0],     # First 5 are 0 (input tokens), next 2 are actual logprobs
    ]
    ```

    All functions receiving or returning GenerationOutputSpec should ensure
    right padding is maintained.
    """

    output_ids: torch.Tensor
    generation_lengths: torch.Tensor  # Length of just the generated response part
    unpadded_sequence_lengths: (
        torch.Tensor
    )  # Length of full valid sequence (input + generated response)
    logprobs: torch.Tensor
    __extra__: Any

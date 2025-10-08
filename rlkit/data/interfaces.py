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
import os
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Optional, Protocol, TypedDict, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rlkit.data.messages import APIMessage

PathLike = Union[str, "os.PathLike[Any]"]
TokenizerType = PreTrainedTokenizerBase

class DatumSpec(TypedDict):
    # GRPO group ID
    idx: int
    
    # Necessary for verifiers rollouts
    prompt: list[APIMessage]
    answer: NotRequired[str] = None
    info: NotRequired[dict[str, Any]] = None
    task: str
    
    # Filled in after rollouts completed
    completion: NotRequired[list[APIMessage]] = None
    reward: NotRequired[float] = None
    
    # Filled in after tokenization
    input_ids: NotRequired[torch.Tensor] = None
    generation_logprobs: NotRequired[torch.Tensor] = None
    token_mask: NotRequired[torch.Tensor] = None
    sample_mask: NotRequired[torch.Tensor] = None
    input_lengths: NotRequired[int] = None
    
    # Filled in after rewards
    advantages: NotRequired[torch.Tensor] = None
    
    # Filled in after logprobs
    prev_logprobs: NotRequired[torch.Tensor] = None
    reference_policy_logprobs: NotRequired[torch.Tensor] = None

class DPODatumSpec(TypedDict):
    message_log_chosen: list[APIMessage]
    message_log_rejected: list[APIMessage]
    length_chosen: int
    length_rejected: int
    loss_multiplier: float
    idx: int


@dataclass
class TaskDataSpec:
    task_name: Optional[str] = None
    # prompt
    prompt_file: Optional[PathLike] = None

    system_prompt_file: Optional[PathLike] = None

    def __post_init__(self) -> None:
        def load_prompt_file(
            prompt_file: Optional[PathLike],
        ) -> Optional[str]:
            """Load prompt from file if it exists, otherwise return as is."""
            if prompt_file is None:
                return None
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Prompt file {prompt_file} not found")

        # Load prompts from files if they exist
        self.system_prompt = load_prompt_file(self.system_prompt_file)
        self.prompt = load_prompt_file(self.prompt_file)

    def copy_defaults(self, from_spec: "TaskDataSpec") -> None:
        """Apply default values from another Task instance for any None attributes."""
        default_attrs = {
            "system_prompt": from_spec.system_prompt,
            "prompt": from_spec.prompt,
        }

        for attr_name, default_value in default_attrs.items():
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, default_value)


class TaskDataProcessFnCallable(Protocol):
    """A callable that processes a loaded datum dictionary into a DatumSpec."""

    def __call__(
        self,
        datum_dict: dict[str, Any],
        task_data_spec: TaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int | None,
        idx: int,
    ) -> DatumSpec:
        raise NotImplementedError("Task data process not implemented")

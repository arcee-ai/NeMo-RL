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
from transformers import PreTrainedTokenizerBase

from rlkit.models.generation.vllm_http.config import HttpVllmConfig

TokenizerType = PreTrainedTokenizerBase


def configure_generation_config(
    config: HttpVllmConfig, tokenizer: TokenizerType, is_eval: bool = False
) -> HttpVllmConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    config["pad_token_id"] = tokenizer.pad_token_id
    if config.get("stop_token_ids") is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # Set skip_tokenizer_init for the HTTP backend
    should_init_tokenizer = is_eval or config.get("stop_strings") is not None
    config["vllm_cfg"]["skip_tokenizer_init"] = not should_init_tokenizer

    return config

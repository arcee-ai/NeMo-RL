"""Functions to convert SFT datasets to the RLKit native format."""
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
import json
from collections.abc import Callable
from typing import Any, cast

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rlkit.config.sft import DatasetType

SFTDataTransformFn = Callable[[PreTrainedTokenizerBase | None, dict], dict]

def _transform_oai(tokenizer: PreTrainedTokenizerBase | None, x: dict) -> dict:
    assert tokenizer is not None, "Tokenizer is required for OpenAI dataset transformation"
    conversation = x["conversations"] if "conversations" in x else x["messages"]
    sample_mask = x.get("sample_mask", 1.0)

    if isinstance(conversation, str):
        conversation = json.loads(conversation)

    has_assistant_messages = any(message["role"] == "assistant" for message in conversation)
    if not has_assistant_messages:
        raise ValueError("No assistant messages found in the conversation!")

    oai_tools = x.get("oai_tools")
    if isinstance(oai_tools, str):
        oai_tools = json.loads(oai_tools)

    tokenized = cast(dict[str, Any], tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        tools=oai_tools
    ))

    input_ids = tokenized["input_ids"]
    token_mask = tokenized["assistant_masks"]
    if 1 not in token_mask:
        raise ValueError("No assistant tokens found in the tokenized conversation - ensure your chat format marks assistant tokens.")
    return {
        "input_ids": torch.tensor(input_ids),
        "token_mask": torch.tensor(token_mask),
        "sample_mask": torch.tensor(sample_mask)
    }

def _sharegpt_to_oai(x: dict) -> dict:
    convo = []
    for message in x.get("conversations", []):
        if message.get("from") == "human":
            convo.append({"role": "user", "content": message.get("value")})
        elif message.get("from") == "gpt":
            convo.append({"role": "assistant", "content": message.get("value")})
        else:
            raise ValueError(f"Unknown message role: {message.get('from')}")
    return {
        "conversations": convo,
        "oai_tools": [],
    }

transformations: dict[str, SFTDataTransformFn] = {
    "axolotl": lambda _, x: {
        "input_ids": torch.tensor(x["input_ids"]),
        "token_mask": torch.tensor([a == b for a, b in zip(x["input_ids"], x["labels"], strict=False)]),
        "sample_mask": torch.tensor(1.0)
    },
    "openai_prompt_completion": lambda tokenizer, x: _transform_oai(
        tokenizer,
        {
            "conversations": x.get("prompt", []) + x.get("completion", []),
            "oai_tools": x.get("tools", [])
        }
    ),
    "sharegpt": lambda tokenizer, x: _transform_oai(
        tokenizer,
        _sharegpt_to_oai(x)
    ),
    "openai": _transform_oai,
}

def transform_sample(sample: dict, dataset_type: DatasetType, tokenizer: PreTrainedTokenizerBase | None) -> dict:
    """Transform a single sample into the RLKit native format."""
    if dataset_type != "native":
        if dataset_type not in transformations:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return transformations[dataset_type](tokenizer, sample)
    return sample

def transform_dataset(dataset: Dataset, dataset_type: DatasetType, tokenizer: PreTrainedTokenizerBase | None, num_proc: int = 8) -> Dataset:
    """Transform a dataset into the RLKit native format."""
    if dataset_type != "native":
        if dataset_type not in transformations:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # get names of new columns to drop old ones
        new_cols = list(transformations[dataset_type](tokenizer, dataset[0]).keys())
        old_cols = list(dataset.column_names)
        drop_cols = [col for col in old_cols if col not in new_cols]

        transform_fn = transformations[dataset_type]
        dataset = cast(Dataset, dataset.map(lambda x: transform_fn(tokenizer, x), num_proc=num_proc, remove_columns=drop_cols))
    return dataset

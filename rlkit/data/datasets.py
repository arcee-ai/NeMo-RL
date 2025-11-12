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
from typing import Any, Callable, Optional, Union

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rlkit.data.interfaces import (
    DatumSpec,
    DPODatumSpec,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from rlkit.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from rlkit.distributed.batched_data_dict import BatchedDataDict

TokenizerType = PreTrainedTokenizerBase


def preference_collate_fn(
    data_batch: list[DPODatumSpec],
) -> BatchedDataDict[Any]:
    """Collate function for preference data training.

    This function separates the chosen and rejected responses to create
    two examples per prompt. The chosen and rejected examples are interleaved
    along the batch dimension, resulting in a batch size of 2 * len(data_batch).

    Args:
        data_batch: List of data samples with message_log_chosen, message_log_rejected, length_chosen, length_rejected, loss_multiplier, idx, and task_name fields.

    Returns:
        BatchedDataDict with message_log, length, loss_multiplier, task_name, and idx fields.
    """
    message_log = []
    length = []
    loss_multiplier = []
    idx = []
    task_names = []
    for datum_spec in data_batch:
        ## interleave chosen and rejected examples
        message_log.append(datum_spec["message_log_chosen"])
        message_log.append(datum_spec["message_log_rejected"])
        length.append(datum_spec["length_chosen"])
        length.append(datum_spec["length_rejected"])
        loss_multiplier.extend([datum_spec["loss_multiplier"]] * 2)
        idx.extend([datum_spec["idx"]] * 2)
        task_names.extend([datum_spec.get("task_name", None)] * 2)
    length_batch: torch.Tensor = torch.tensor(length)
    loss_multiplier_batch: torch.Tensor = torch.tensor(loss_multiplier)

    batch_max_length = torch.ones_like(length_batch) * length_batch.max()

    batch: BatchedDataDict[Any] = BatchedDataDict(
        message_log=message_log,
        length=length_batch,
        loss_multiplier=loss_multiplier_batch,
        task_name=task_names,
        idx=idx,
        batch_max_length=batch_max_length,
    )

    return batch

# Dataset transformation utility
SFTDataTransformFn = Callable[[PreTrainedTokenizerBase | None, dict], dict]

def transform_oai(tokenizer: PreTrainedTokenizerBase | None, x: dict) -> dict:
    assert tokenizer is not None, "Tokenizer is required for OpenAI dataset transformation"
    conversation = x["conversations"]
    sample_mask = x.get("sample_mask", 1.0)
    
    if isinstance(conversation, str):
        conversation = json.loads(conversation)
    
    has_assistant_messages = any(message["role"] == "assistant" for message in conversation)
    if not has_assistant_messages:
        raise ValueError("No assistant messages found in the conversation!")
    
    oai_tools = x.get("oai_tools", None)
    if isinstance(oai_tools, str):
        oai_tools = json.loads(oai_tools)
    
    tokenized = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        tools=oai_tools
    )
        
    input_ids = tokenized["input_ids"]
    token_mask = tokenized["assistant_masks"]
    if 1 not in token_mask:
        raise ValueError("No assistant tokens found in the tokenized conversation - ensure your chat format marks assistant tokens.")
    return {
        "input_ids": torch.tensor(input_ids),
        "token_mask": torch.tensor(token_mask),
        "sample_mask": torch.tensor(sample_mask)
    }

def sharegpt_to_oai(x: dict) -> dict:
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
        "token_mask": torch.tensor([a == b for a, b in zip(x["input_ids"], x["labels"])]),
        "sample_mask": torch.tensor(1.0)
    },
    "openai_prompt_completion": lambda tokenizer, x: transform_oai(
        tokenizer,
        {
            "conversations": x.get("prompt", []) + x.get("completion", []),
            "oai_tools": x.get("tools", [])
        }
    ),
    "sharegpt": lambda tokenizer, x: transform_oai(
        tokenizer,
        sharegpt_to_oai(x)
    ),
    "openai": transform_oai,
}

def transform_dataset(dataset: Dataset, dataset_type: str, tokenizer: PreTrainedTokenizerBase | None, num_proc: int = 8) -> Dataset:
    if dataset_type != "native":
        if dataset_type not in transformations:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # get names of new columns to drop old ones
        new_cols = list(transformations[dataset_type](tokenizer, dataset[0]).keys())
        old_cols = list(dataset.column_names)
        drop_cols = [col for col in old_cols if col not in new_cols]
        
        transform_fn = transformations[dataset_type]
        dataset = dataset.map(lambda x: transform_fn(tokenizer, x), num_proc=num_proc, remove_columns=drop_cols)
    return dataset
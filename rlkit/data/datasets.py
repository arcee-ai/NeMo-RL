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
from typing import Any, Optional, Union

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
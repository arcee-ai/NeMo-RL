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

import torch

from rlkit.data.datasets import preference_collate_fn
from rlkit.data.interfaces import DatumSpec
from rlkit.distributed.batched_data_dict import BatchedDataDict


def test_preference_collate_fn():
    """Test that preference_collate_fn correctly processes preference training data."""
    data_batch = [
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "token_ids": torch.tensor([4, 5, 6, 7]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Bye",
                    "token_ids": torch.tensor([8, 9]),
                },
            ],
            length_chosen=7,
            length_rejected=5,
            loss_multiplier=1.0,
            idx=0,
            task_name="test_task",
        ),
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "I'm good",
                    "token_ids": torch.tensor([13, 14, 15]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "Not great",
                    "token_ids": torch.tensor([16, 17, 18, 19]),
                },
            ],
            length_chosen=6,
            length_rejected=7,
            loss_multiplier=0,
            idx=1,
            task_name="test_task",
        ),
    ]

    train_data = preference_collate_fn(data_batch)

    # Verify the output structure
    assert isinstance(train_data, BatchedDataDict)
    assert "message_log" in train_data
    assert "length" in train_data
    assert "loss_multiplier" in train_data
    assert "task_name" in train_data
    assert "idx" in train_data
    assert "batch_max_length" in train_data

    # Verify batch size is doubled (chosen + rejected for each example)
    assert len(train_data["message_log"]) == 4  # 2 examples * 2 (chosen + rejected)

    # Verify lengths are interleaved chosen/rejected
    expected_lengths = torch.tensor([7, 5, 6, 7])
    assert torch.equal(train_data["length"], expected_lengths)

    # Verify loss multipliers duplicated for chosen/rejected
    expected_loss_multipliers = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert torch.equal(train_data["loss_multiplier"], expected_loss_multipliers)

    # Verify idx values duplicated for chosen/rejected samples
    assert train_data["idx"] == [0, 0, 1, 1]

    # Verify task names duplicated for chosen/rejected samples
    assert train_data["task_name"] == ["test_task"] * 4

    # Verify batch_max_length matches the maximum sequence length across the batch
    assert torch.equal(train_data["batch_max_length"], torch.tensor([7, 7, 7, 7]))

    # Verify message logs for first example are preserved
    first_chosen = train_data["message_log"][0]
    first_rejected = train_data["message_log"][1]
    assert torch.equal(first_chosen[0]["token_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(first_chosen[1]["token_ids"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(first_rejected[0]["token_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(first_rejected[1]["token_ids"], torch.tensor([8, 9]))

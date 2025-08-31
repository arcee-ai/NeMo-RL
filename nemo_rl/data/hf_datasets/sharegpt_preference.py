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

from typing import Any

from datasets import DatasetDict, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


def _convert_conversations(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in conv:
        role_raw = str(turn.get("from", "")).lower()
        role = ROLE_MAP.get(role_raw, role_raw)
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    return messages


def to_preference_data_format(example: dict[str, Any]) -> dict[str, Any]:
    """Convert ShareGPT-style preference JSON into the common preference format.

    Input example structure:
        {
            "conversations": [{"from": "system"|"user"|"assistant", "value": str}, ...],
            "chosen": str,
            "rejected": str,
        }

    Output structure:
        {
            "context": list[{"role": str, "content": str}],
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": str}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": str}]},
            ],
        }
    """
    messages = _convert_conversations(example["conversations"])  # type: ignore[index]
    chosen_suffix = example.get("chosen", "")
    rejected_suffix = example.get("rejected", "")

    if messages and messages[-1]["role"] == "assistant":
        base = messages[-1].get("content", "")
        context = messages[:-1]
        chosen_text = f"{base}{chosen_suffix}"
        rejected_text = f"{base}{rejected_suffix}"
    else:
        context = messages
        chosen_text = chosen_suffix
        rejected_text = rejected_suffix

    return {
        "context": context,
        "completions": [
            {"rank": 0, "completion": [{"role": "assistant", "content": chosen_text}]},
            {"rank": 1, "completion": [{"role": "assistant", "content": rejected_text}]},
        ],
    }


class ShareGptPreferenceDataset:
    """Preference dataset loader for ShareGPT-style JSON files.

    Expects `train_ds_path` and `val_ds_path` to point to JSON files with the
    structure described in `to_preference_data_format`.
    """

    def __init__(self, train_ds_path: str, val_ds_path: str) -> None:
        train_raw = load_dataset("json", data_files=train_ds_path)["train"]
        val_raw = load_dataset("json", data_files=val_ds_path)["train"]

        formatted_train = train_raw.map(to_preference_data_format)
        formatted_val = val_raw.map(to_preference_data_format)

        self.formatted_ds = DatasetDict(
            {
                "train": formatted_train,
                "validation": formatted_val,
            }
        )

        self.task_spec = TaskDataSpec(task_name="ShareGptPreference")

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

"""Contains data processors for evaluation."""

from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec

TokenizerType = PreTrainedTokenizerBase


def hf_math_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process chat-formatted math samples into a DatumSpec.

    This matches the behaviour of the legacy GRPO math script, keeping support for
    datasets that provide `messages` with user and assistant turns.
    """

    messages = datum_dict.get("messages", [])
    if len(messages) < 2:
        raise ValueError("Expected at least user and assistant messages in datum.")

    user_problem = messages[0]["content"]
    ground_truth = messages[1]["content"]

    prompt_template = getattr(task_data_spec, "prompt", None)
    if prompt_template:
        user_problem = prompt_template.format(user_problem)

    user_message = {"role": "user", "content": user_problem}
    formatted_message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        formatted_message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = formatted_message

    message_log: LLMMessageLogType = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {"ground_truth": ground_truth},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def _construct_multichoice_prompt(
    prompt: str, question: str, options: dict[str, str]
) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output


def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(sys, return_tensors="pt")["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        question = _construct_multichoice_prompt(
            task_data_spec.prompt, question, options
        )
    user_message = {"role": "user", "content": question}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

"""Test the SFT dataset transformations."""
import json

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from rlkit.data.sft_datasets import transform_dataset, transform_sample


def test_oai():
    """Test OpenAI dataset transform."""
    tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Trinity-Mini")
    sample = {
        "conversations": [
            {"role": "user", "content": "Hello, how are you?"},
            {
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "New York"})
                    }
                ]
            }
        ],
        "oai_tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            }
        ]
    }

    tokenized = tokenizer.apply_chat_template(
        sample["conversations"],
        tools=sample["oai_tools"],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True
    )

    transformed = transform_sample(sample, "openai", tokenizer)

    assert (transformed["input_ids"] == torch.tensor(tokenized["input_ids"])).all()
    assert (transformed["token_mask"] == torch.tensor(tokenized["assistant_masks"])).all()
    assert transformed["sample_mask"] == 1.0

def test_sharegpt():
    """Test ShareGPT dataset transform."""
    tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Trinity-Mini")

    sample_oai = {
        "conversations": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
    }

    sample_sharegpt = {
        "conversations": [
            {"from": "human", "value": "Hello, how are you?"},
            {"from": "gpt", "value": "I'm doing well, thank you!"}
        ]
    }

    transformed = transform_sample(sample_sharegpt, "sharegpt", tokenizer)

    tokenized = tokenizer.apply_chat_template(
        sample_oai["conversations"],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True
    )

    assert (transformed["input_ids"] == torch.tensor(tokenized["input_ids"])).all()
    assert (transformed["token_mask"] == torch.tensor(tokenized["assistant_masks"])).all()
    assert transformed["sample_mask"] == 1.0

def test_dataset():
    """Test dataset transformation."""
    dataset = Dataset.from_list([
        {
            "conversations": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": f"I'm doing well, thank you! The number is {i}"}
            ]
        }
        for i in range(10)
    ])

    tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Trinity-Mini")
    transformed = transform_dataset(dataset, "openai", tokenizer)
    assert len(transformed) == 10

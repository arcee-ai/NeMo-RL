from dataclasses import dataclass
from typing import Any

@dataclass
class DataConfig:
    max_input_seq_length: int
    dataset_name: str
    prompt_file: str | None = None
    system_prompt_file: str | None = None
    val_dataset_name: str | None = None
    add_bos: bool | None = None
    add_eos: bool | None = None
    input_key: str | None = None
    output_key: str | None = None
    add_generation_prompt: bool | None = None
    add_system_prompt: bool | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    split: str | None = None
    shuffle: bool | None = None

from typing import Any, TypedDict, NotRequired

class DataConfig(TypedDict):
    max_input_seq_length: int
    dataset_name: str
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    val_dataset_name: NotRequired[str | None]
    add_bos: NotRequired[bool | None]
    add_eos: NotRequired[bool | None]
    input_key: NotRequired[str | None]
    output_key: NotRequired[str | None]
    add_generation_prompt: NotRequired[bool | None]
    add_system_prompt: NotRequired[bool | None]
    tokenizer_kwargs: NotRequired[dict[str, Any] | None]
    split: NotRequired[str | None]
    shuffle: NotRequired[bool | None]

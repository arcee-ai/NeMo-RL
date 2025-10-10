from typing import Any, Literal, TypedDict, NotRequired

class DataConfig(TypedDict):
    dataset_name: str
    dataset_type: NotRequired[Literal["axolotl", "native"]]
    on_disk: NotRequired[bool]
    shuffle: NotRequired[bool | None]

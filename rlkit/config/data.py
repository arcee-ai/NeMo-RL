"""Configuration for datasets."""
from typing import Literal, TypedDict, NotRequired

class DataConfig(TypedDict):
    """Configuration for datasets."""
    dataset_name: str
    dataset_type: NotRequired[Literal["axolotl", "native"]]
    on_disk: NotRequired[bool]
    shuffle: NotRequired[bool | None]

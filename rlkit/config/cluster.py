"""Configuration for the number of available GPUs."""
from typing import TypedDict

class ClusterConfig(TypedDict):
    """Configuration for the number of available GPUs."""
    gpus_per_node: int
    num_nodes: int
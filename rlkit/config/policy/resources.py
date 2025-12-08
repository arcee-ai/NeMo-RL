"""Configuration for resources."""

from pydantic import BaseModel


class ResourcesConfig(BaseModel):
    """Configuration for how many GPUs are allocated to something."""

    gpus_per_node: int
    num_nodes: int

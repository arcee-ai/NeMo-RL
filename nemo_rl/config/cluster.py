from typing import TypedDict

class ClusterConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int
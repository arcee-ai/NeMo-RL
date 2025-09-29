from dataclasses import dataclass

@dataclass
class ClusterConfig:
    gpus_per_node: int
    num_nodes: int
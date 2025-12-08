"""vLLM server coordinator."""
from rlkit.config.policy import PolicyConfig
import os
import sys
import time
from typing import Any

import openai
import ray

from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.inference.vllm_http import VLLMOpenAIServe

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


class VllmHttpGeneration:
    """vLLM server coordinator."""
    def __init__(self, cluster: RayVirtualCluster, config: PolicyConfig):
        """Initialize the vLLM server coordinator."""
        # Save config for later use
        self.cfg = config

        assert config.inference is not None, "RL requires an inference configuration."

        runtime_env = {
            "py_executable": sys.executable,
            "env_vars": {**os.environ, "VLLM_USE_V1": "1", "NCCL_CUMEM_ENABLE": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
        }

        # Use Ray Serve replicas for data parallelism, and keep vLLM's internal DP at 1.
        self.tp_size = config.inference.tp_size

        # Calculate DP size from total GPUs and TP/PP size
        self.num_nodes = config.inference.resources.num_nodes
        gpus_per_node = config.inference.resources.gpus_per_node
        gpus_per_node = 1 if gpus_per_node is None else int(gpus_per_node)
        self.dp_size = gpus_per_node // self.tp_size

        self.actors = []

        # Get list of available Ray nodes to schedule actors on different nodes
        # Filter to only alive nodes with GPUs
        all_nodes = ray.nodes() # type: ignore[arg-type] - pyrefly hallucinates a fake param here
        available_nodes = [
            n for n in all_nodes
            if n.get("Alive", False) and n.get("Resources", {}).get("GPU", 0) > 0
        ]

        if len(available_nodes) < self.num_nodes:
            raise RuntimeError(
                f"Not enough nodes with GPUs available. Need {self.num_nodes} nodes, "
                f"but only {len(available_nodes)} nodes with GPUs are available."
            )

        # Get node IDs for scheduling
        node_ids = [n["NodeID"] for n in available_nodes[:self.num_nodes]]
        print(f"Scheduling {self.num_nodes} vLLM actors across nodes: {node_ids}")

        # Create all actors in parallel - each is on a different node so no GPU competition
        server_timeout = config.inference.server_timeout

        for i in range(self.num_nodes):
            # Use NodeAffinitySchedulingStrategy to force this actor to a specific node
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_ids[i],
                soft=False,  # Hard constraint - must be on this node
            )

            actor = VLLMOpenAIServe.options( # type: ignore - type checking strangeness w/r/t Ray options method
                name=f"vllm_http_generation_{i}",
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=config.model_name,
                tensor_parallel_size=self.tp_size,
                pipeline_parallel_size=1,
                max_model_len=config.max_total_sequence_length,
                gpu_memory_utilization=config.inference.gpu_memory_utilization,
                data_parallel_size=self.dp_size,
            )
            self.actors.append(actor)

        print(f"Created {self.num_nodes} vLLM actors, waiting for engines to initialize...")

        # Wait for all actors to finish initializing in parallel
        polling_start = time.time()
        initialized = [False] * self.num_nodes

        while time.time() - polling_start < server_timeout:
            # Check all uninitialized actors
            for i, actor in enumerate(self.actors):
                if initialized[i]:
                    continue
                try:
                    engine_ready = ray.get(actor.admin_engine_ready.remote(), timeout=2.0)
                    if engine_ready:
                        initialized[i] = True
                        print(f"vLLM actor {i+1}/{self.num_nodes} initialized")
                except Exception:
                    pass

            if all(initialized):
                break
            time.sleep(1)

        if not all(initialized):
            failed = [i+1 for i, ok in enumerate(initialized) if not ok]
            raise RuntimeError(f"vLLM actors {failed} did not initialize in time (waited {server_timeout} seconds)")

        print(f"All {self.num_nodes} vLLM actors initialized successfully")

        self.client = openai.OpenAI(api_key="n/a", base_url="http://127.0.0.1:8000/v1")
        # The served model name from VLLMOpenAIServe defaults to "policy"
        self.served_model_name = "policy"

    def get_ips(self) -> list[str]:
        """Get a list of IP addresses for the vLLM servers."""
        return ray.get([actor.admin_get_ip.remote() for actor in self.actors])

    def init_collective(self, ip: str, port: int, world_size: int) -> list[ray.ObjectRef]:
        """Initialize collective communicaton between vLLM workers and the training rank0 for weight refits.

        Args:
            ip: The IP address of the head node.
            port: The port to use for collective communication.
            world_size: The total number of workers in the collective (train rank0 + inference workers).

        Returns:
            list[ray.ObjectRef]: Futures to await alongside training futures for setting up collective communication.
        """
        return [
            actor.admin_init_collective.remote(i * self.dp_size * self.tp_size, ip, port, world_size)
            for i, actor in enumerate(self.actors)
        ]

    async def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Reset prefix cache on all vLLM servers."""
        ray.get([actor.admin_reset_prefix_cache.remote() for actor in self.actors])
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare vLLM workers for weight refits."""
        ray.get([actor.admin_prepare_refit_info.remote(state_dict_info) for actor in self.actors])

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Update weights from collective communication.

        Returns:
            list[ray.ObjectRef]: Futures to await alongside training futures for updating weights from collective communication.
        """
        return [actor.admin_update_from_collective.remote() for actor in self.actors]

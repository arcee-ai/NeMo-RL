"""Wrapper class for all of the training workers."""
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue
from transformers import PreTrainedTokenizerBase

from rlkit.algorithms.loss_functions import LossFunction
from rlkit.distributed.named_sharding import NamedSharding
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from rlkit.config import PolicyConfig
from rlkit.training.utils import get_device_mesh_info

PathLike = Union[str, "os.PathLike[Any]"]


class Policy:
    """Wrapper class for all of the training workers."""
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "lm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[PathLike] = None,
        optimizer_path: Optional[PathLike] = None,
        init_reference_model: bool = True,
        use_hf_checkpoint: bool = False,
        use_cut_cross_entropy: bool = False,
    ):
        """Initialize all training workers."""
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        dtensor_v2_cfg = config["dtensor_v2_cfg"]
        
        assert dtensor_v2_cfg is not None, "DTensorV2Config is required"

        worker_builder_cls = "rlkit.models.policy.v2_policy_worker.DTensorV2PolicyWorker"
        tp_size = dtensor_v2_cfg.get("tensor_parallel_size", 1)
        cp_size = dtensor_v2_cfg.get("context_parallel_size", 1)
        pp_size = dtensor_v2_cfg.get("pipeline_parallel_size", 1)
        ep_size = dtensor_v2_cfg.get("expert_parallel_size", 1)
        dp_replicate = dtensor_v2_cfg.get("dp_replicate", 1)
        env_vars = dtensor_v2_cfg.get("env_vars", {})

        # Build a flattened DP axis for DTensorV2: dp = dp_replicate * dp_shard_mod_ep * dp_shard_in_ep
        mesh_info = get_device_mesh_info(
            cluster.world_size(),
            tp_size,
            cp_size,
            ep_size,
            pp_size,
            dp_replicate,
            always_include_all=True,
        )

        shape_map = {n: s for n, s in zip(mesh_info["mesh_dim_names"], mesh_info["mesh_shape"])}
        dp_axis_size = (
            shape_map.get("dp_replicate", 1)
            * shape_map.get("dp_shard_mod_ep", 1)
            * shape_map.get("dp_shard_in_ep", 1)
        )
        new_shape = [
            shape_map.get("pp", 1),
            max(1, dp_axis_size),
            shape_map.get("cp", 1),
            shape_map.get("tp", 1),
        ]
        new_names = [
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ]

        assert np.prod(new_shape) == cluster.world_size(), (
            f"NamedSharding shape {tuple(new_shape)} product "
            f"!= world_size {cluster.world_size()}"
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(*new_shape),
            names=new_names,
        )

        pre_init_queue = RayQueue()
        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            tokenizer=tokenizer,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
            use_hf_checkpoint=use_hf_checkpoint,
            use_cut_cross_entropy=use_cut_cross_entropy,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            workers_per_node=workers_per_node,
            sharding_annotations=self.sharding_annotations,
            env_vars=env_vars,
        )

        self.cfg = config
        self.use_hf_checkpoint = use_hf_checkpoint

    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication.
        
        Args:
            ip: The IP address of the head node.
            port: The port to use for collective communication.
            world_size: The total number of workers in the collective (train rank0 + inference workers).
        
        Returns:
            list[ray.ObjectRef]: Futures to await alongside vLLM futures for setting up collective communication.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "init_collective", ip=ip, port=port, world_size=world_size
        )
        return futures

    async def train(
        self,
        sharded_data: list[list[dict[str, list[int | float]]]],
        loss_fn: LossFunction,
        pad_values: dict[str, int | float | bool],
        eval_mode: bool = False,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function.
        
        Args:
            sharded_data: List of shards (one per DP rank), where each shard is a list
                of packed samples (dicts with token_ids, token_mask, etc.).
            loss_fn: Loss function to use for training.
            pad_values: Dictionary mapping field names to the placeholder value to use when padding tensors.
            eval_mode: Whether to run in evaluation mode (no gradient updates).
        
        Returns:
            dict[str, Any]: A dictionary containing the metrics from the training step.
        """
        assert len(sharded_data) > 0, "Data must contain at least one shard"
        
        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_sharded_data(
            "train",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "pad_values": pad_values,
            },
        )
        results = await self.worker_group.get_all_worker_results_async(futures)

        # Aggregate the results
        aggregated_results = {
            "loss": results[0]["loss"],
            "grad_norm": results[0]["grad_norm"],
        }

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        # Aggregate router statistics across DP ranks
        # Router statistics are already aggregated across EP ranks within each worker
        # We need to sum absolute counts across DP ranks, then convert to fractions
        router_stats_all_workers = []
        for r in results:
            if "router_statistics" in r:
                router_stats_all_workers.append(r["router_statistics"])
        
        if router_stats_all_workers:
            # Sum absolute counts across all DP ranks
            aggregated_router_stats = {}
            # Get all expert keys from first worker (should be same across all)
            if router_stats_all_workers:
                expert_keys = router_stats_all_workers[0].keys()
                # Group by layer to normalize fractions per layer
                layer_stats = {}
                for expert_key in expert_keys:
                    # Parse layer_id from expert_key (format: "expert_{layer_id}_{expert_idx}")
                    parts = expert_key.split("_")
                    if len(parts) >= 3:
                        layer_id = "_".join(parts[1:-1])  # Handle multi-digit layer IDs
                        expert_idx = parts[-1]
                        layer_key = f"layer_{layer_id}"
                        if layer_key not in layer_stats:
                            layer_stats[layer_key] = {}
                        layer_stats[layer_key][expert_idx] = sum(
                            stats.get(expert_key, 0) for stats in router_stats_all_workers
                        )
                
                # Convert to fractions per layer and calculate expert balance
                for layer_key, expert_counts in layer_stats.items():
                    total_counts = sum(expert_counts.values())
                    layer_id = layer_key.replace("layer_", "")
                    expert_fractions = []
                    
                    if total_counts > 0:
                        for expert_idx, count in expert_counts.items():
                            # Reconstruct expert key
                            expert_key = f"expert_{layer_id}_{expert_idx}"
                            fraction = count / total_counts
                            aggregated_router_stats[expert_key] = fraction
                            expert_fractions.append(fraction)
                    else:
                        # If no tokens routed, set all fractions to 0
                        for expert_idx in expert_counts.keys():
                            expert_key = f"expert_{layer_id}_{expert_idx}"
                            aggregated_router_stats[expert_key] = 0.0
                            expert_fractions.append(0.0)
                    
                    # Calculate expert balance metric (standard deviation of expert fractions)
                    # Lower values indicate better balance (more even distribution)
                    if len(expert_fractions) > 1:
                        import numpy as np
                        expert_balance = np.std(expert_fractions)
                    else:
                        # Single expert case - perfect balance by definition
                        expert_balance = 0.0
                    aggregated_router_stats[f"expert_balance_{layer_id}"] = expert_balance
            aggregated_results["router_statistics"] = aggregated_router_stats

        return aggregated_results

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Prepares the workers for a training step, onloading everything to the GPU."""
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def prepare_refit_info(self) -> dict[str, Any]:
        """Prepare the info for refit.

        Returns:
            dict[str, Any]: A dictionary containing the info for refit.
        """
        futures = self.worker_group.run_all_workers_single_data("prepare_refit_info")
        results = ray.get(futures)
        # Only get the first worker's info since all workers will have the same result
        return results[0]

    def broadcast_weights_for_collective(self) -> list[ray.ObjectRef]:
        """Start futures for broadcasting model weights to inference workers.
        
        These futures should be awaited alongside the vLLM futures for receiving the weight.
        
        Returns:
            list[ray.ObjectRef]: Futures to await alongside vLLM futures for broadcasting model weights.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective"
        )
        return futures

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            tokenizer_path=tokenizer_path,
        )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all HF workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("start_gpu_profiling")
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("stop_gpu_profiling")
        ray.get(futures)

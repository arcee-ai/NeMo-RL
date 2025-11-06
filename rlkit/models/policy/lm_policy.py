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
import asyncio
import os
import uuid
import warnings
from collections import defaultdict
from typing import Any, Optional, Union
import logging

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue
from transformers import PreTrainedTokenizerBase

from rlkit.algorithms.interfaces import LossFunction
from rlkit.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
    SlicedDataDict,
)
from rlkit.distributed.named_sharding import NamedSharding
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from rlkit.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from rlkit.config import PolicyConfig
from rlkit.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from rlkit.utils.flops_tracker import (
    FLOPTracker,
    get_default_hf_config,
    get_theoretical_tflops,
)
from rlkit.models.policy.v2_policy_worker import get_device_mesh_info

PathLike = Union[str, "os.PathLike[Any]"]


class Policy(ColocatablePolicyInterface, GenerationInterface):
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
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder_cls: str
        tp_size = 1
        pp_size = 1
        cp_size = 1

        dtv2_enable = config.get("dtensor_v2_cfg", {}).get("enabled", False)
        dtensor_cfg = config.get("dtensor_cfg", {}) or {}
        dtensor_enable = dtensor_cfg.get("enabled", False)

        if dtv2_enable:
            worker_builder_cls = (
                "rlkit.models.policy.v2_policy_worker.DTensorV2PolicyWorker"
            )
            tp_size = config["dtensor_v2_cfg"].get("tensor_parallel_size", 1)
            cp_size = config["dtensor_v2_cfg"].get("context_parallel_size", 1)
            pp_size = config["dtensor_v2_cfg"].get("pipeline_parallel_size", 1)
            ep_size = config["dtensor_v2_cfg"].get("expert_parallel_size", 1)
            env_vars = config["dtensor_v2_cfg"].get("env_vars", {})
        else:
            assert dtensor_enable, (
                "Please set policy.dtensor_cfg.enabled=true to use the DTensor training backend "
                "or set policy.torchtitan_cfg.enabled=true to use the TorchTitan backend."
            )
            worker_builder_cls = (
                "rlkit.models.policy.dtensor_policy_worker.DTensorPolicyWorker"
            )
            tp_size = dtensor_cfg.get("tensor_parallel_size", 1)
            cp_size = dtensor_cfg.get("context_parallel_size", 1)
            env_vars = dtensor_cfg.get("env_vars", {})

        if dtv2_enable:
            dp_size = cluster.world_size() // (tp_size * cp_size * pp_size * ep_size)
            # Build a flattened DP axis for DTensorV2: dp = dp_replicate * dp_shard_mod_ep * dp_shard_in_ep
            mesh_info = get_device_mesh_info(
                cluster.world_size(),
                tp_size,
                cp_size,
                ep_size,
                pp_size,
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
        else:
            self.sharding_annotations = NamedSharding(
                layout=np.arange(cluster.world_size()).reshape(
                    pp_size,  # PP
                    -1,  # DP
                    cp_size,  # CP
                    tp_size,  # TP
                ),
                names=[
                    "pipeline_parallel",
                    "data_parallel",
                    "context_parallel",
                    "tensor_parallel",
                ],
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

        if config["dynamic_batching"]["enabled"]:
            assert pp_size == 1, (
                "Dynamic batching is only supported for single pipeline parallel stage"
            )
            self.use_dynamic_batches = True
            self.dynamic_batching_args: DynamicBatchingArgs = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": config["dynamic_batching"][
                    "sequence_length_round"
                ],
                "max_tokens_per_microbatch": 0,  # Override this in each different call (presumably different sizes)
            }
            assert not config["sequence_packing"]["enabled"], (
                "Dynamic Batching is exclusive of Sequence Packing. Please disable Sequence Packing to use Dynamic Batching"
            )
        else:
            self.use_dynamic_batches = False

        # initialize FLOPs tracker
        try:
            self.flops_tracker = FLOPTracker.from_config(
                config["model_name"], get_default_hf_config(config["model_name"])
            )
        except ValueError as e:
            self.flops_tracker = None
            print(f"FLOPS tracker not supported for model {config['model_name']}: {e}")

        if config["sequence_packing"]["enabled"]:
            self.use_sequence_packing = True
            self.sequence_packing_args: SequencePackingArgs = {
                "train_mb_tokens": config["sequence_packing"]["train_mb_tokens"],
                "logprob_mb_tokens": config["sequence_packing"].get(
                    "logprob_mb_tokens", None
                ),
                "algorithm": config["sequence_packing"]["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": (cp_size * 2 * tp_size)
                if cp_size > 1
                else tp_size,
            }
            assert not config["dynamic_batching"]["enabled"], (
                "Sequence Packing is exclusive of Dynamic Batching. Please disable Dynamic Batching"
            )
        else:
            self.use_sequence_packing = False

        self.cfg = config
        self.use_hf_checkpoint = use_hf_checkpoint

    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "init_collective", ip=ip, port=port, world_size=world_size
        )
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    async def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]

        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        elif self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["logprob_mb_tokens"]
            # we just shard into DP shards here as Sequence packing allows for CP.
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                dp_size,
                batch_size=None,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
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
        )
        worker_results = await self.worker_group.get_all_worker_results_async(futures)
        logprobs: BatchedDataDict[LogprobOutputSpec] = BatchedDataDict.from_batches(
            worker_results
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        elif self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                dp_size,
                batch_size=None,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
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
            common_kwargs={"micro_batch_size": micro_batch_size},
        )
        logprobs: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            BatchedDataDict.from_batches(
                self.worker_group.get_all_worker_results(futures)
            )
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    async def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        # Shard and replicate the batch
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["train_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(
                dp_size,
                batch_size=batch_size,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        elif self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["train_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(
                dp_size,
                batch_size=batch_size,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                dp_size,
                batch_size=batch_size,
            )

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
            for shard in sharded_data:
                input_lengths = shard["input_lengths"]
                self.flops_tracker.track_batch(input_lengths.tolist())

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
                "gbs": batch_size,
                "mbs": micro_batch_size,
            },
        )
        results = await self.worker_group.get_all_worker_results_async(futures)

        # Aggregate the results
        aggregated_results = {
            "loss": results[0]["global_loss"],
            "grad_norm": results[0]["grad_norm"],
        }

        if self.flops_tracker is not None:
            aggregated_results["total_flops"] = self.flops_tracker.total_flops
            aggregated_results["num_ranks"] = len(results)

            try:
                aggregated_results["theoretical_tflops"] = sum(
                    get_theoretical_tflops(r["gpu_name"], r["model_dtype"])
                    for r in results
                )
            except Exception as e:
                warnings.warn(f"Error getting theoretical flops: {e}")

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

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        # Verify input data is right-padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"greedy": greedy},
        )
        assert self.cfg["generation"] is not None, "Generation config is not set"
        result: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["pad_token_id"]},
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference"
        )
        ray.get(futures)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass

    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare the info for refit.

        Returns:
            dict: A dictionary containing the info for refit.
        """
        futures = self.worker_group.run_all_workers_single_data("prepare_refit_info")
        results = ray.get(futures)
        # Only get the first worker's info since all workers will have the same result
        return results[0]

    def prepare_weights_for_ipc(
        self, _refit_buffer_size_gb: Optional[int] = None
    ) -> list[list[str]]:
        """Prepare the weights for IPC.

        Returns:
            list: A list containing the keys of the parameters, which is grouped by size.
        """
        # Get the state_dict_info and available memory from all workers
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_weights_for_ipc"
        )
        results = ray.get(futures)

        # Only get the first worker's state_dict_info since all workers will have the same result
        state_dict_info = results[0][0]

        if _refit_buffer_size_gb is not None:
            total_available_bytes = _refit_buffer_size_gb * (1024**3)
        else:
            # Get the minimum available memory from all workers
            total_available_bytes = min(result[1] for result in results)

        # Group tensors by size
        cur_available_bytes = total_available_bytes
        grouped_param_keys: list[list[str]] = []
        keys: list[str] = []

        for key, size_in_bytes in state_dict_info:
            if size_in_bytes > cur_available_bytes:
                if keys:
                    grouped_param_keys.append(keys)
                    keys = []
                cur_available_bytes = total_available_bytes

            keys.append(key)
            cur_available_bytes -= size_in_bytes

        if keys:
            grouped_param_keys.append(keys)

        return grouped_param_keys

    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles: list[dict[str, Any]] = ray.get(
            [
                worker.get_weights_ipc_handles.remote(keys=keys)
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def broadcast_weights_for_collective(self) -> list[ray.ObjectRef]:
        """Broadcast the weights for collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective"
        )
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_before_refit")
        ray.get(futures)

    def offload_after_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_after_refit")
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        if not self.use_hf_checkpoint:
            self._verify_dcp_checkpoint_dir(weights_path)

        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            tokenizer_path=tokenizer_path,
        )
        ray.get(futures)

    def _verify_dcp_checkpoint_dir(self, weights_path: str) -> None:
        """Ensure the checkpoint directory is visible from every worker before DCP save."""
        if weights_path is None:
            raise ValueError("weights_path must be provided when using DCP checkpoints.")
        target_dir = os.path.abspath(weights_path)
        os.makedirs(target_dir, exist_ok=True)
        marker_path = os.path.join(
            target_dir,
            f".dcp_fs_check_{uuid.uuid4().hex}",
        )
        # Create the marker file on the driver/head node
        with open(marker_path, "wb"):
            pass

        try:
            futures = self.worker_group.run_all_workers_single_data(
                "check_file_visibility",
                path=marker_path,
            )
            visibility = ray.get(futures)
        finally:
            try:
                os.remove(marker_path)
            except FileNotFoundError:
                pass

        missing_workers = [idx for idx, seen in enumerate(visibility) if not seen]
        if missing_workers:
            raise RuntimeError(
                "Distributed checkpoint directory is not shared across all workers. "
                f"Marker file {marker_path} was not visible on worker indices {missing_workers}. "
                "Ensure the checkpoint directory resides on shared storage or enable HF checkpoints."
            )

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

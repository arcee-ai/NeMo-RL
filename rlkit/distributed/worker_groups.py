"""Classes to manage Ray worker groups."""
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
import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rlkit.distributed.named_sharding import NamedSharding
from rlkit.distributed.virtual_cluster import RayVirtualCluster

logger = logging.getLogger(__name__)


@dataclass
class WorkerFutures:
    """Container for Ray futures with worker selection info for result filtering."""

    futures: list[ray.ObjectRef]
    return_from_workers: list[int] = field(default_factory=list)
    called_workers: list[int] = field(default_factory=list)


class RayWorkerBuilder:
    """Builder for Ray workers."""

    def __init__(self, worker_class_fqn: str, *args: Any, **kwargs: Any):
        """Initialize the builder with a worker class and its constructor arguments."""
        self.worker_class_fqn = worker_class_fqn
        self.args = args
        self.kwargs = kwargs
        self._worker_class: type | None = None

    @property
    def worker_class(self) -> type:
        """Lazily import and cache the worker class."""
        if self._worker_class is None:
            module_name, class_name = self.worker_class_fqn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self._worker_class = getattr(module, class_name)
        return self._worker_class

    def create(
        self,
        placement_group: PlacementGroup,
        bundle_index: int,
        num_gpus: float,
        bundle_indices: tuple[int, list[int]] | None = None,
        **extra_options: Any,
    ) -> ray.actor.ActorHandle:
        """Create a Ray worker actor.

        Args:
            placement_group: Ray placement group for resource allocation.
            bundle_index: Index of the bundle in the placement group.
            num_gpus: Number of GPUs to allocate.
            bundle_indices: Optional (node_idx, bundle_indices) for model ownership.
            **extra_options: Additional Ray actor options.

        Returns:
            Ray actor handle for the created worker.
        """
        options: dict[str, Any] = dict(extra_options)

        # Apply worker-specific configuration if available
        worker_kwargs = dict(self.kwargs)
        if hasattr(self.worker_class, "configure_worker"):
            resources, env_vars, init_kwargs = self.worker_class.configure_worker(
                num_gpus=num_gpus,
                bundle_indices=bundle_indices,
            )
            if resources and "num_gpus" in resources:
                num_gpus = resources["num_gpus"]
            if env_vars:
                options.setdefault("runtime_env", {}).setdefault("env_vars", {}).update(env_vars)
            if init_kwargs:
                worker_kwargs.update(init_kwargs)

        # Set scheduling strategy
        options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=bundle_index,
            placement_group_capture_child_tasks=True,
        )
        options["num_gpus"] = num_gpus

        return self.worker_class.options(**options).remote(*self.args, **worker_kwargs) # type: ignore[attr-defined] - pyrefly doesn't understand Ray


class RayWorkerGroup:
    """Manages distributed Ray workers for parallel execution.

    Handles worker creation, placement, environment setup, and provides
    methods to execute tasks across workers with sharding support.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        worker_builder: RayWorkerBuilder,
        workers_per_node: int | list[int] | None = None,
        name_prefix: str = "",
        sharding_annotations: NamedSharding | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        """Initialize a group of distributed Ray workers.

        Args:
            cluster: Virtual cluster providing placement groups.
            worker_builder: Builder for creating Ray worker actors.
            workers_per_node: Number of workers per node. Defaults to one per bundle.
            name_prefix: Prefix for worker names.
            sharding_annotations: NamedSharding for multi-dimensional worker arrangement.
            env_vars: Environment variables to pass to workers.
        """
        self.cluster = cluster
        self.name_prefix = name_prefix
        self.sharding_annotations = sharding_annotations
        self._workers: list[ray.actor.ActorHandle] = []
        self._worker_metadata: list[dict[str, Any]] = []

        env_vars = env_vars or {}
        placement_groups = cluster.get_placement_groups()

        # Determine workers per placement group
        if workers_per_node is None:
            workers_per_group = [pg.bundle_count for pg in placement_groups]
        elif isinstance(workers_per_node, int):
            workers_per_group = [workers_per_node] * len(placement_groups)
        else:
            workers_per_group = workers_per_node

        # Build flat list of (pg_idx, bundle_idx) for all workers
        worker_placements = [
            (pg_idx, bundle_idx)
            for pg_idx, count in enumerate(workers_per_group)
            for bundle_idx in range(count)
        ]

        self._create_workers(worker_builder, worker_placements, env_vars)

    def _create_workers(
        self,
        worker_builder: RayWorkerBuilder,
        worker_placements: list[tuple[int, int]],
        env_vars: dict[str, str],
    ) -> None:
        """Create all workers."""
        master_addr, master_port = self.cluster.get_master_address_and_port()
        placement_groups = self.cluster.get_placement_groups()
        world_size = len(worker_placements)

        # Merge with current environment
        base_env = {k: v for k, v in os.environ.items() if k not in env_vars}
        base_env.update(env_vars)

        logger.info(f"Creating {world_size} workers...")

        for global_rank, (pg_idx, bundle_idx) in enumerate(worker_placements):
            pg = placement_groups[0] if len(placement_groups) == 1 else placement_groups[pg_idx]

            worker_env = {
                **base_env,
                "RANK": str(global_rank),
                "LOCAL_RANK": str(bundle_idx),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "NODE_RANK": str(pg_idx),
            }
            worker_env.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)

            runtime_env = {
                "env_vars": {"VIRTUAL_ENV": sys.executable, "UV_PROJECT_ENVIRONMENT": sys.executable, **worker_env},
                "py_executable": sys.executable,
            }

            num_gpus = 1 / self.cluster.max_colocated_worker_groups if self.cluster.use_gpus else 0
            name = f"{self.name_prefix}-{pg_idx}-{bundle_idx}"

            # Only rank 0 of each DP shard gets bundle_indices (for model ownership)
            bundle_indices = (pg_idx, [bundle_idx]) if global_rank == 0 or self._is_dp_leader(global_rank) else None

            worker = worker_builder.create(
                placement_group=pg,
                bundle_index=bundle_idx,
                num_gpus=num_gpus,
                bundle_indices=bundle_indices,
                runtime_env=runtime_env,
                name=name,
            )

            self._workers.append(worker)
            self._worker_metadata.append({
                "node_idx": pg_idx,
                "local_rank": bundle_idx,
                "global_rank": global_rank,
                "name": name,
                "bundle_indices": bundle_indices,
            })

        # Wait for all workers to be ready
        logger.info(f"Waiting for {world_size} workers to initialize...")
        ray.get([w.__ray_ready__.remote() for w in self._workers])

    def _is_dp_leader(self, worker_idx: int) -> bool:
        """Check if worker is a data parallel leader (first in its DP group)."""
        if self.sharding_annotations is None:
            return worker_idx == 0
        coords = self.sharding_annotations.get_worker_coords(worker_idx)
        # Leader if all non-DP coordinates are 0
        for axis in self.sharding_annotations.names:
            if axis == "data_parallel":
                continue
            if coords.get(axis, 0) != 0:
                return False
        return True

    @property
    def workers(self) -> list[ray.actor.ActorHandle]:
        """List of Ray actor handles for all workers."""
        return self._workers

    def run_all(self, method_name: str, **kwargs: Any) -> list[ray.ObjectRef]:
        """Run a method on all workers with the same arguments.

        Args:
            method_name: Method to call on each worker.
            **kwargs: Arguments to pass to the method.

        Returns:
            List of Ray futures for each worker's result.
        """
        return [getattr(w, method_name).remote(**kwargs) for w in self._workers]

    def run_sharded(
        self,
        method_name: str,
        sharded_axes: list[str] | None = None,
        replicate_axes: list[str] | None = None,
        output_replicated_axes: list[str] | None = None,
        common_kwargs: dict[str, Any] | None = None,
        **sharded_kwargs: Any,
    ) -> WorkerFutures:
        """Run a method with data sharded/replicated across worker dimensions.

        Args:
            method_name: Method to call on each worker.
            sharded_axes: Axes along which data is already sharded (send slice to each).
            replicate_axes: Axes along which to replicate data (send same to all).
            output_replicated_axes: Axes along which output is replicated (return from rank 0 only).
            common_kwargs: Arguments to pass to all workers unchanged.
            **sharded_kwargs: Data to shard. Each value should be indexable by axis coordinates.

        Returns:
            WorkerFutures with futures and metadata for result filtering.
        """
        if self.sharding_annotations is None:
            raise ValueError("Sharding annotations required for run_sharded")

        sharded_axes = sharded_axes or []
        replicate_axes = replicate_axes or []
        output_replicated_axes = output_replicated_axes or []
        common_kwargs = common_kwargs or {}

        # Validate axes
        all_axes = set(sharded_axes + replicate_axes)
        for axis in all_axes:
            if axis not in self.sharding_annotations.names:
                raise ValueError(f"Unknown axis '{axis}'. Valid: {self.sharding_annotations.names}")
        if set(sharded_axes) & set(replicate_axes):
            raise ValueError("Axes cannot be both sharded and replicated")

        futures = []
        called_workers = []
        return_from_workers = []

        for worker_idx, worker in enumerate(self._workers):
            coords = self.sharding_annotations.get_worker_coords(worker_idx)

            # Skip workers not at rank 0 on "free" axes (not sharded or replicated)
            is_free_axis_nonzero = any(
                coords.get(axis, 0) != 0
                for axis in self.sharding_annotations.names
                if axis not in sharded_axes and axis not in replicate_axes
            )
            if is_free_axis_nonzero:
                continue

            # Determine if we should return results from this worker
            should_return = all(
                coords.get(axis, 0) == 0
                for axis in self.sharding_annotations.names
                if axis in output_replicated_axes or (axis not in sharded_axes and axis not in replicate_axes)
            )
            if should_return:
                return_from_workers.append(worker_idx)

            # Select the appropriate data slice for sharded axes
            worker_kwargs = dict(sharded_kwargs)
            for axis in sharded_axes:
                if axis in coords:
                    worker_kwargs = {k: v[coords[axis]] for k, v in worker_kwargs.items()}

            future = getattr(worker, method_name).remote(**worker_kwargs, **common_kwargs)
            futures.append(future)
            called_workers.append(worker_idx)

        return WorkerFutures(
            futures=futures,
            called_workers=called_workers,
            return_from_workers=return_from_workers,
        )

    async def get_results_async(self, worker_futures: WorkerFutures) -> list[Any]:
        """Get results from worker futures, filtering to non-replicated outputs.

        Args:
            worker_futures: WorkerFutures from run_sharded().

        Returns:
            List of results, one per non-replicated output.
        """
        if not worker_futures.futures:
            return []

        all_results = ray.get(list(worker_futures.futures))

        if not worker_futures.return_from_workers:
            return all_results

        # Map called workers to result indices
        worker_to_idx = {w: i for i, w in enumerate(worker_futures.called_workers)}
        return [
            all_results[worker_to_idx[w]]
            for w in worker_futures.return_from_workers
            if w in worker_to_idx
        ]

    def shutdown(self, cleanup_method: str | None = None, timeout: float = 30.0, force: bool = False) -> bool:
        """Shutdown all workers.

        Args:
            cleanup_method: Optional method to call on workers before termination.
            timeout: Timeout for graceful shutdown.
            force: Force kill even if cleanup_method is provided.

        Returns:
            True if all workers shut down successfully.
        """
        if not self._workers:
            return True

        success = True

        # Graceful shutdown
        if cleanup_method and not force:
            try:
                futures = self.run_all(cleanup_method)
                ray.get(futures, timeout=timeout)
            except (ray.exceptions.RayTaskError, ray.exceptions.GetTimeoutError) as e:
                logger.warning(f"Graceful shutdown failed: {e}. Force killing...")
                force = True
                success = False

        # Force kill
        if force or not cleanup_method:
            for worker in self._workers:
                try:
                    ray.kill(worker)
                except Exception as e:
                    logger.error(f"Error killing worker: {e}")
                    success = False

        self._workers = []
        self._worker_metadata = []
        return success

"""Ray + vLLM Python API backend."""
import asyncio
import contextlib
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@dataclass
class VllmGenerationResult:
    """Minimal generation result for rollout stitching."""

    completion_token_ids: list[int]
    completion_logprobs: list[float]
    text: str
    finish_reason: str | None


def _collect_request_output(generator, aggregate: bool):
    async def _collect():
        final = None
        async for output in generator:
            if final is None or not aggregate:
                final = output
            else:
                final.add(output, aggregate=True)
        return final

    return _collect()


def _extract_chosen_logprobs(logprobs_obj, token_ids: list[int]) -> list[float]:
    logprobs: list[float] = []
    for i, token_id in enumerate(token_ids):
        token_logprobs = logprobs_obj[i]
        if not isinstance(token_logprobs, dict):
            token_logprobs = dict(token_logprobs)
        if token_id not in token_logprobs:
            raise ValueError(f"Missing logprob for token {token_id} at position {i}")
        logprobs.append(token_logprobs[token_id].logprob)
    return logprobs


@ray.remote(num_cpus=1)
class VllmRayActor:
    """Single vLLM engine hosted inside a Ray actor."""

    def __init__(
        self,
        model: str,
        engine_args: dict[str, Any],
        worker_extension_cls: str,
    ):
        """Create the vLLM engine inside this Ray actor."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self._engine_initialized = False
        self._engine = None

        args = dict(engine_args)
        args["model"] = model
        args["worker_extension_cls"] = worker_extension_cls
        engine_args_obj = AsyncEngineArgs(**args)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args_obj)
        self._engine_initialized = True

    async def admin_engine_ready(self) -> bool:
        """Report whether the vLLM engine finished initialization."""
        return bool(self._engine_initialized)

    async def admin_get_ip(self) -> str:
        """Return the node IP for this actor."""
        return ray.util.get_node_ip_address()

    async def admin_reset_prefix_cache(self) -> bool:
        """Clear vLLM prefix cache on all workers."""
        assert self._engine is not None
        await self._engine.collective_rpc("reset_prefix_cache", args=())
        return True

    async def admin_init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> bool:
        """Initialize the distributed collective for refit operations."""
        assert self._engine is not None
        await self._engine.collective_rpc(
            "init_collective", args=(rank_prefix, ip, port, world_size)
        )
        return True

    async def admin_prepare_refit_info(self, state_dict_info: dict) -> bool:
        """Prepare refit metadata on workers before weight updates."""
        assert self._engine is not None
        await self._engine.collective_rpc("prepare_refit_info", args=(state_dict_info,))
        return True

    async def admin_update_from_collective(self) -> bool:
        """Apply refit updates from the collective."""
        assert self._engine is not None
        results = await self._engine.collective_rpc(
            "update_weights_from_collective", args=()
        )
        return bool(results and results[0])

    async def admin_report_device_id(self) -> list[str]:
        """Return device identifiers for each worker."""
        assert self._engine is not None
        return await self._engine.collective_rpc("report_device_id", args=())

    async def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params_dict: dict[str, Any],
        request_id: str,
    ) -> VllmGenerationResult:
        """Run a single vLLM generation request."""
        from vllm.inputs.data import TokensPrompt
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        assert self._engine is not None
        sampling_params = SamplingParams(**sampling_params_dict)

        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        generator = self._engine.generate(prompt, sampling_params, request_id)
        aggregate = sampling_params.output_kind == RequestOutputKind.DELTA
        output = await _collect_request_output(generator, aggregate)

        if output is None or not output.outputs:
            raise RuntimeError("vLLM returned no outputs")

        completion = output.outputs[0]
        completion_ids = list(completion.token_ids)
        if completion.logprobs is None:
            raise RuntimeError("vLLM returned no logprobs for completion")
        completion_logprobs = _extract_chosen_logprobs(
            completion.logprobs, completion_ids
        )

        result = VllmGenerationResult(
            completion_token_ids=completion_ids,
            completion_logprobs=completion_logprobs,
            text=completion.text,
            finish_reason=completion.finish_reason,
        )

        return result


class VllmRayBackend:
    """Ray-backed pool of vLLM engines."""

    def __init__(
        self,
        model: str,
        max_model_len: int,
        inference_config,
        worker_extension_cls: str = "rlkit.inference.worker_ext.VllmHttpWorkerExtension",
    ):
        """Create a Ray-backed pool of vLLM actors."""
        self.cfg = inference_config
        self.tp_size = int(self.cfg.tp_size)
        self.pp_size = int(self.cfg.pp_size)
        self.gpus_per_actor = self.tp_size * self.pp_size
        if self.gpus_per_actor <= 0:
            raise ValueError("tp_size and pp_size must be >= 1")

        gpus_per_node = int(self.cfg.resources.gpus_per_node)
        num_nodes = int(self.cfg.resources.num_nodes)

        if gpus_per_node % self.gpus_per_actor != 0:
            raise ValueError(
                f"gpus_per_node ({gpus_per_node}) must be divisible by gpus_per_actor ({self.gpus_per_actor})"
            )

        actors_per_node = gpus_per_node // self.gpus_per_actor
        if actors_per_node == 0:
            raise ValueError(
                f"Not enough GPUs per node for tp_size*pp_size={self.gpus_per_actor}"
            )

        self.num_nodes = num_nodes
        self.dp_size = actors_per_node * num_nodes
        self._actors: list[ray.actor.ActorHandle] = []
        self._placement_groups = []
        self._inflight = []

        engine_args = {
            "tensor_parallel_size": self.tp_size,
            "pipeline_parallel_size": self.pp_size,
            "enable_expert_parallel": bool(self.cfg.enable_expert_parallel),
            "max_model_len": int(max_model_len),
            "gpu_memory_utilization": float(self.cfg.gpu_memory_utilization),
            "dtype": self.cfg.dtype,
            "distributed_executor_backend": "mp",
            "data_parallel_size": 1,
            "max_num_seqs": int(self.cfg.max_num_seqs),
            "max_num_batched_tokens": self.cfg.max_num_batched_tokens,
            "enable_chunked_prefill": bool(self.cfg.enable_chunked_prefill),
            "disable_log_stats": bool(self.cfg.disable_log_stats),
            "trust_remote_code": True,
            "logprobs_mode": "processed_logprobs",
        }
        engine_args.update(self.cfg.extra_engine_args)

        runtime_env = {
            "env_vars": {
                **os.environ,
                "VLLM_USE_V1": "1",
                "NCCL_CUMEM_ENABLE": "1",
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        }

        startup_timeout_s = int(self.cfg.startup_timeout_s)

        for _ in range(self.dp_size):
            pg = placement_group(
                bundles=[{"CPU": 1, "GPU": 1}] * self.gpus_per_actor,
                strategy="PACK",
            )
            self._placement_groups.append(pg)

        try:
            ray.get([pg.ready() for pg in self._placement_groups], timeout=startup_timeout_s)
        except (TimeoutError, ray.exceptions.GetTimeoutError) as e:
            for pg in self._placement_groups:
                with contextlib.suppress(Exception):
                    remove_placement_group(pg)
            raise TimeoutError(
                "Timed out waiting for vLLM placement groups to be ready."
            ) from e

        for i, pg in enumerate(self._placement_groups):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            actor = VllmRayActor.options(
                name=f"vllm_ray_actor_{i}",
                num_gpus=self.gpus_per_actor,
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=model,
                engine_args=engine_args,
                worker_extension_cls=worker_extension_cls,
            )
            self._actors.append(actor)
            self._inflight.append(0)

        self._wait_for_ready(startup_timeout_s)

    def _wait_for_ready(self, timeout_s: int) -> None:
        start = time.time()
        initialized = [False] * len(self._actors)
        while time.time() - start < timeout_s:
            for i, actor in enumerate(self._actors):
                if initialized[i]:
                    continue
                try:
                    if ray.get(actor.admin_engine_ready.remote(), timeout=2.0):
                        initialized[i] = True
                except Exception:
                    pass
            if all(initialized):
                return
            time.sleep(1)
        missing = [i for i, ok in enumerate(initialized) if not ok]
        raise RuntimeError(f"vLLM actors not ready: {missing}")

    def _choose_actor_index(self) -> int:
        if len(self._actors) == 1:
            return 0
        a, b = random.sample(range(len(self._actors)), 2)
        return a if self._inflight[a] <= self._inflight[b] else b

    async def generate(
        self, prompt_token_ids: list[int], sampling_params_dict: dict[str, Any]
    ) -> VllmGenerationResult:
        """Dispatch a generation request to a selected actor."""
        idx = self._choose_actor_index()
        self._inflight[idx] += 1
        try:
            request_id = f"rlkit_{time.time_ns()}"
            result = await self._actors[idx].generate.remote(
                prompt_token_ids=prompt_token_ids,
                sampling_params_dict=sampling_params_dict,
                request_id=request_id,
            )
            return result
        finally:
            self._inflight[idx] -= 1

    async def finish_generation(self) -> bool:
        """Finalize any pending generation state before training."""
        await asyncio.gather(
            *[actor.admin_reset_prefix_cache.remote() for actor in self._actors]
        )
        return True

    def init_collective(self, ip: str, port: int, world_size: int) -> list[ray.ObjectRef]:
        """Initialize the collective across all actors."""
        futures = []
        for i, actor in enumerate(self._actors):
            rank_prefix = i * self.gpus_per_actor
            futures.append(actor.admin_init_collective.remote(rank_prefix, ip, port, world_size))
        return futures

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Send refit metadata to every actor."""
        ray.get([actor.admin_prepare_refit_info.remote(state_dict_info) for actor in self._actors])

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Trigger refit weight updates on all actors."""
        return [actor.admin_update_from_collective.remote() for actor in self._actors]

    def get_ips(self) -> list[str]:
        """Return node IPs for all actors."""
        return ray.get([actor.admin_get_ip.remote() for actor in self._actors])

    def shutdown(self) -> None:
        """Release placement groups associated with this backend."""
        for pg in self._placement_groups:
            with contextlib.suppress(Exception):
                remove_placement_group(pg)

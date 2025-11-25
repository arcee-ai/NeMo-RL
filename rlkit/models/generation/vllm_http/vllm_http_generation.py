import time
import os
import asyncio
from typing import Any, Optional, cast, AsyncGenerator

import openai
import ray
import requests
from requests.exceptions import RequestException
import torch
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.ray_actor_environment_registry import get_actor_python_env
from rlkit.utils.venvs import create_local_venv_on_each_node
from rlkit.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)

from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.models.generation.vllm_http.config import HttpVllmConfig
from rlkit.models.generation.vllm_http.vllm_http import VLLMOpenAIServe

from ray import serve
from ray.serve.handle import DeploymentHandle


class VllmHttpGeneration(GenerationInterface):
    client: openai.OpenAI | openai.AsyncOpenAI
    
    def __init__(self, cluster: RayVirtualCluster, config: HttpVllmConfig):
        # Save config for later use
        self.cfg = config

        # serve.start(detached=False, http_options={"port": 8000, "host": "127.0.0.1", "location": "EveryNode", "access_log": False})
    
        py_exec = get_actor_python_env("rlkit.models.generation.vllm_http.vllm_http.VLLMOpenAIServe")

        # Resolve uv-based executables to a concrete venv python under venvs/
        if isinstance(py_exec, str) and py_exec.startswith("uv"):
            py_exec = create_local_venv_on_each_node(
                py_executable=py_exec,
                venv_name="rlkit.models.generation.vllm_http.vllm_http.VLLMOpenAIServe",
            )

        # Prepare runtime_env to ensure Serve replicas use the managed venv
        runtime_env = {"py_executable": py_exec, "env_vars": {}}
        try:
            venv_dir = os.path.dirname(os.path.dirname(py_exec)) if isinstance(py_exec, str) else None
            if venv_dir:
                runtime_env["env_vars"] = {
                    "VIRTUAL_ENV": venv_dir,
                    "UV_PROJECT_ENVIRONMENT": venv_dir,
                }
        except Exception:
            pass
        
        # TODO: find better place for this, force V1 engine
        runtime_env["env_vars"]["VLLM_USE_V1"] = "1"
        runtime_env["env_vars"]["NCCL_CUMEM_ENABLE"] = "1"
        # TODO: I really don't like this. Find a way around torch dtype serialization.
        runtime_env["env_vars"]["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Use Ray Serve replicas for data parallelism, and keep vLLM's internal DP at 1.
        self.tp_size = config["vllm_cfg"]["tensor_parallel_size"]
        self.pp_size = config["vllm_cfg"]["pipeline_parallel_size"]
        
        # Calculate DP size from total GPUs and TP/PP size
        num_nodes = config["colocated"]["resources"].get("num_nodes", 1)
        num_nodes = 1 if num_nodes is None else int(num_nodes)
        gpus_per_node = config["colocated"]["resources"].get("gpus_per_node", 1)
        gpus_per_node = 1 if gpus_per_node is None else int(gpus_per_node)
        self.dp_size = gpus_per_node // (self.tp_size * self.pp_size)

        self.actors = []
        
        for i in range(num_nodes):
            self.actors.append(VLLMOpenAIServe.options(name=f"vllm_http_generation_{i}", runtime_env=runtime_env).remote(
                model=config["model_name"],
                tensor_parallel_size=self.tp_size,
                pipeline_parallel_size=self.pp_size,
                max_model_len=config["vllm_cfg"]["max_model_len"],
                gpu_memory_utilization=config["vllm_cfg"]["gpu_memory_utilization"],
                data_parallel_size=self.dp_size,
            ))
        
        # serve.run(vllm_app, route_prefix="/", name="vllm_http_generation")
        
        # Poll vLLM server until it's ready to avoid race condition
        print("Waiting for vLLM server to come online...")
        polling_start = time.time()
        success = False
        
        server_timeout = config.get("server_timeout", 60)
        while time.time() - polling_start < server_timeout:
            try:
                response = requests.get("http://127.0.0.1:8000/v1/models")
                if response.status_code == 200:
                    success = True
                    break
                time.sleep(1)
            except RequestException:
                pass
        
        if not success:
            raise RuntimeError("vLLM server did not come online in time (waited {} seconds)".format(server_timeout))

        print(f"vLLM server is online at http://127.0.0.1:8000/v1")

        self.client = openai.OpenAI(api_key="n/a", base_url="http://127.0.0.1:8000/v1")
        # The served model name from VLLMOpenAIServe defaults to "policy"
        self.served_model_name = "policy"

    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        raise NotImplementedError("The vLLM HTTP generation backend is only for use with verifiers environments.")

    async def generate_async(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict["GenerationOutputSpec"]], None]:
        raise NotImplementedError("The vLLM HTTP generation backend is only for use with verifiers environments.")

    def shutdown(self):
        serve.shutdown()

    def init_collective(self, ip: str, port: int, world_size: int):
        return [actor.admin_init_collective.remote(0, ip, port, world_size) for actor in self.actors]

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    async def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # Wait for the reset to complete across replicas
        ray.get([actor.admin_reset_prefix_cache.remote() for actor in self.actors])
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        # Wait for refit prep to complete across replicas.
        ray.get([actor.admin_prepare_refit_info.remote(state_dict_info) for actor in self.actors])

    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        raise NotImplementedError("update_weights_from_ipc_handles is not supported for vLLM over HTTP")

    def update_weights_from_collective(self):
        return [actor.admin_update_from_collective.remote() for actor in self.actors]
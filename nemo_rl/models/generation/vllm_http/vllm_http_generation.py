import time
import os
import asyncio
from typing import Any, Optional, cast, AsyncGenerator

import openai
import ray
import requests
from requests.exceptions import RequestException
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.utils.venvs import create_local_venv_on_each_node
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.vllm_http.config import HttpVllmConfig
from nemo_rl.models.generation.vllm_http.vllm_http import VLLMOpenAIServe

from ray import serve


class VllmHttpGeneration(GenerationInterface):
    client: openai.OpenAI | openai.AsyncOpenAI
    
    def __init__(self, cluster: RayVirtualCluster, config: HttpVllmConfig):
        # Save config for later use
        self.cfg = config

        serve.start(detached=False, http_options={"port": 8000, "host": "127.0.0.1", "location": "EveryNode"})
    
        py_exec = get_actor_python_env("nemo_rl.models.generation.vllm_http.vllm_http.VLLMOpenAIServe")

        # Resolve uv-based executables to a concrete venv python under venvs/
        if isinstance(py_exec, str) and py_exec.startswith("uv"):
            py_exec = create_local_venv_on_each_node(
                py_executable=py_exec,
                venv_name="nemo_rl.models.generation.vllm_http.vllm_http.VLLMOpenAIServe",
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

        vllm_app = VLLMOpenAIServe.options( # type: ignore
            ray_actor_options={
                "num_cpus": 1,
                # "num_gpus": config["colocated"]["resources"]["gpus_per_node"],
                "runtime_env": runtime_env,
            }
        ).bind(
            model=config["model_name"],
            tensor_parallel_size=config["vllm_cfg"]["tensor_parallel_size"],
            max_model_len=config["vllm_cfg"]["max_model_len"],
            extra_cli_args=config["vllm_cfg"]["extra_cli_args"]
        )
        
        serve.run(vllm_app, route_prefix="/", name="vllm_http_generation")
        
        # Poll vLLM server until it's ready to avoid race condition
        print("Waiting for vLLM server to come online...")
        polling_start = time.time()
        success = False
        while time.time() - polling_start < config["server_timeout"]:
            try:
                response = requests.get("http://127.0.0.1:8000/v1/models")
                if response.status_code == 200:
                    success = True
                    break
                time.sleep(1)
            except RequestException:
                pass
        
        if not success:
            raise RuntimeError("vLLM server did not come online in time (waited {} seconds)".format(config["server_timeout"]))

        print(f"vLLM server is online at http://127.0.0.1:8000/v1")

        self.client = openai.OpenAI(api_key="n/a", base_url="http://127.0.0.1:8000/v1")
        # The served model name from VLLMOpenAIServe defaults to "policy"
        self.served_model_name = "policy"

    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        # Validate required inputs
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for generation"
        )

        input_ids: torch.Tensor = data["input_ids"]
        input_lengths: torch.Tensor = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])

        batch_size = input_ids.shape[0]
        padded_input_length = input_ids.size(1)

        max_new_tokens: int = self.cfg["max_new_tokens"]
        temperature: float = 0.0 if greedy else self.cfg["temperature"]
        top_p: float = self.cfg["top_p"]
        top_k: Optional[int] = 1 if greedy else (self.cfg["top_k"] if self.cfg["top_k"] is not None else -1)

        # Prepare per-sample requests (use vLLM OpenAI extension: prompt_token_ids)
        generated_token_id_lists: list[list[int]] = []
        generated_logprobs_lists: list[list[float]] = []
        max_generated = 0

        for i in range(batch_size):
            valid_len = int(input_lengths[i].item())
            prompt_token_ids = input_ids[i, :valid_len].tolist() if valid_len > 0 else []
            stop_strings_i: Optional[list[str]] = None
            if batch_stop_strings and i < len(batch_stop_strings):
                stop_strings_i = batch_stop_strings[i]

            gen_token_ids, gen_logprobs = self._generate_one(
                prompt_token_ids=prompt_token_ids,
                stop_strings=stop_strings_i,
                greedy=greedy,
            )

            generated_token_id_lists.append(gen_token_ids)
            generated_logprobs_lists.append(gen_logprobs)
            if len(gen_token_ids) > max_generated:
                max_generated = len(gen_token_ids)

        # Assemble batched outputs with right padding
        pad_id = self.cfg.get("pad_token_id")
        assert pad_id is not None, "pad_token_id must be provided in config"

        total_length = padded_input_length + max_generated

        output_ids_list: list[torch.Tensor] = []
        logprobs_list: list[torch.Tensor] = []
        generation_lengths: list[int] = []
        unpadded_sequence_lengths: list[int] = []

        for i in range(batch_size):
            seq_len = int(input_lengths[i].item())
            gen_ids = generated_token_id_lists[i]
            gen_lps = generated_logprobs_lists[i]

            full_output = torch.full((total_length,), pad_id, dtype=input_ids.dtype)
            full_output[:seq_len] = input_ids[i][:seq_len]
            if len(gen_ids) > 0:
                full_output[seq_len : seq_len + len(gen_ids)] = torch.tensor(gen_ids, dtype=input_ids.dtype)

            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            for j, lp in enumerate(gen_lps):
                pos = seq_len + j
                if pos < total_length:
                    full_logprobs[pos] = float(lp)
            logprobs_list.append(full_logprobs)

            generation_lengths.append(len(gen_ids))
            unpadded_sequence_lengths.append(seq_len + len(gen_ids))

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
                "unpadded_sequence_lengths": torch.tensor(unpadded_sequence_lengths, dtype=torch.long),
            }
        )

    def _generate_one(
        self,
        *,
        prompt_token_ids: list[int],
        stop_strings: Optional[list[str]],
        greedy: bool,
    ) -> tuple[list[int], list[float]]:
        max_new_tokens: int = self.cfg["max_new_tokens"]
        temperature: float = 0.0 if greedy else self.cfg["temperature"]
        top_p: float = self.cfg["top_p"]
        top_k: Optional[int] = 1 if greedy else (self.cfg["top_k"] if self.cfg["top_k"] is not None else -1)

        resp = self.client.completions.create(
            model=self.served_model_name,
            prompt=prompt_token_ids,
            # max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_strings if stop_strings else None,
            logprobs=True
        )

        choice = resp.choices[0]
        gen_token_ids: list[int] = []
        gen_logprobs: list[float] = []

        content = getattr(choice.logprobs, "content", None)
        if content is not None:
            for token_info in content:
                token_id = getattr(token_info, "token_id", None)
                if token_id is None:
                    top_list = getattr(token_info, "top_logprobs", None)
                    if top_list and len(top_list) > 0:
                        token_id = getattr(top_list[0], "token_id", None)
                        logprob = getattr(top_list[0], "logprob", 0.0)
                    else:
                        logprob = 0.0
                else:
                    logprob = getattr(token_info, "logprob", 0.0)

                if token_id is None:
                    raise RuntimeError(
                        "vLLM HTTP server did not return token_id in logprobs; enable processed_logprobs with token ids"
                    )
                gen_token_ids.append(int(token_id))
                gen_logprobs.append(float(logprob))
        else:
            raise RuntimeError(
                "vLLM HTTP response lacks token-level logprobs; cannot construct token ids"
            )

        return gen_token_ids, gen_logprobs

    async def generate_async(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict["GenerationOutputSpec"]], None]:
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for generation"
        )
        input_ids: torch.Tensor = data["input_ids"]
        input_lengths: torch.Tensor = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])

        batch_size = input_ids.shape[0]
        pad_id = self.cfg.get("pad_token_id")
        assert pad_id is not None, "pad_token_id must be provided in config"

        loop = asyncio.get_running_loop()

        tasks: list[asyncio.Task] = []
        indices: list[int] = []
        for i in range(batch_size):
            valid_len = int(input_lengths[i].item())
            prompt_token_ids = input_ids[i, :valid_len].tolist() if valid_len > 0 else []
            stop_strings_i: Optional[list[str]] = None
            if batch_stop_strings and i < len(batch_stop_strings):
                stop_strings_i = batch_stop_strings[i]

            task = asyncio.create_task(
                asyncio.to_thread(
                    self._generate_one,
                    prompt_token_ids=prompt_token_ids,
                    stop_strings=stop_strings_i,
                    greedy=greedy,
                )
            )
            tasks.append(task)
            indices.append(i)

        # Yield samples as they complete
        for completed in asyncio.as_completed(tasks):
            gen_ids, gen_lps = await completed
            # Find original index for this task
            # Map by popping first not-yet-done; maintain parallel order using tasks list
            idx = tasks.index(cast(asyncio.Task, completed))
            original_idx = indices[idx]

            seq_len = int(input_lengths[original_idx].item())
            padded_input_length = input_ids.size(1)
            total_length = seq_len + len(gen_ids)

            full_output = torch.full((total_length,), pad_id, dtype=input_ids.dtype)
            if seq_len > 0:
                full_output[:seq_len] = input_ids[original_idx][:seq_len]
            if len(gen_ids) > 0:
                full_output[seq_len : seq_len + len(gen_ids)] = torch.tensor(gen_ids, dtype=input_ids.dtype)

            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            for j, lp in enumerate(gen_lps):
                pos = seq_len + j
                if pos < total_length:
                    full_logprobs[pos] = float(lp)

            single = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": full_output.unsqueeze(0),
                    "logprobs": full_logprobs.unsqueeze(0),
                    "generation_lengths": torch.tensor([len(gen_ids)], dtype=torch.long),
                    "unpadded_sequence_lengths": torch.tensor([seq_len + len(gen_ids)], dtype=torch.long),
                }
            )

            yield original_idx, single

    def get_deployment_handle(self):
        return serve.get_deployment_handle("VLLMOpenAIServe", app_name="vllm_http_generation")

    # The following interface methods are no-ops for now
    def init_collective(self, ip: str, port: int, world_size: int):
        h = self.get_deployment_handle()
        return [h.admin_init_collective.remote(0, ip, port, world_size)]

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        h = self.get_deployment_handle()
        # Wait for the reset to complete
        if self.cfg["vllm_cfg"]["async_engine"]:
            h.admin_reset_prefix_cache_async.remote().result()
        else:
            h.admin_reset_prefix_cache.remote().result()
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        h = self.get_deployment_handle()
        # Wait for refit prep to complete.
        h.admin_prepare_refit_info.remote(state_dict_info).result()

    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        raise NotImplementedError("update_weights_from_ipc_handles is not supported for vLLM over HTTP")

    def update_weights_from_collective(self):
        h = self.get_deployment_handle()
        return [h.admin_update_from_collective.remote()]
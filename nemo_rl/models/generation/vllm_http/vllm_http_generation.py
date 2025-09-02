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
from nemo_rl.models.custom.model import BaseModelArgs
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
from ray.serve.handle import DeploymentHandle


class VllmHttpGeneration(GenerationInterface):
    client: openai.OpenAI | openai.AsyncOpenAI
    
    def __init__(self, cluster: RayVirtualCluster, config: HttpVllmConfig):
        # Save config for later use
        self.cfg = config

        serve.start(detached=False, http_options={"port": 8000, "host": "127.0.0.1", "location": "EveryNode", "access_log": False})
    
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
        # TODO: bodge to force NCCL version match
        runtime_env["env_vars"]["VLLM_NCCL_SO_PATH"] = f"{os.path.dirname(torch.__file__)}/lib/libnccl.so.2"

        # Use Ray Serve replicas for data parallelism, and keep vLLM's internal DP at 1.
        self.tp_size = config["vllm_cfg"]["tensor_parallel_size"]
        self.pp_size = config["vllm_cfg"]["pipeline_parallel_size"]
        
        # Calculate DP size from total GPUs and TP/PP size
        num_nodes = config["colocated"]["resources"].get("num_nodes", 1)
        num_nodes = 1 if num_nodes is None else int(num_nodes)
        gpus_per_node = config["colocated"]["resources"].get("gpus_per_node", 1)
        gpus_per_node = 1 if gpus_per_node is None else int(gpus_per_node)
        total_gpus = num_nodes * gpus_per_node
        self.dp_size = total_gpus // (self.tp_size * self.pp_size)

        vllm_app = VLLMOpenAIServe.options( # type: ignore
            ray_actor_options={
                "num_cpus": 1,
                "runtime_env": runtime_env,
            },
        ).bind(
            model=config["model_name"],
            tensor_parallel_size=self.tp_size,
            pipeline_parallel_size=self.pp_size,
            max_model_len=config["vllm_cfg"]["max_model_len"],
            gpu_memory_utilization=config["vllm_cfg"]["gpu_memory_utilization"],
            data_parallel_size=self.dp_size,
            extra_cli_args=config["vllm_cfg"].get("extra_cli_args", [])
        )
        
        serve.run(vllm_app, route_prefix="/", name="vllm_http_generation")
        
        # Poll vLLM server until it's ready to avoid race condition
        print("Waiting for vLLM server to come online...")
        # polling_start = time.time()
        # success = False
        
        # server_timeout = config.get("server_timeout", 60)
        # while time.time() - polling_start < server_timeout:
        #     try:
        #         response = requests.get("http://127.0.0.1:8000/v1/models")
        #         if response.status_code == 200:
        #             success = True
        #             break
        #         time.sleep(1)
        #     except RequestException:
        #         pass
        
        # if not success:
        #     raise RuntimeError("vLLM server did not come online in time (waited {} seconds)".format(server_timeout))

        print(f"vLLM server is online at http://127.0.0.1:8000/v1")

        self.client = openai.OpenAI(api_key="n/a", base_url="http://127.0.0.1:8000/v1")
        # The served model name from VLLMOpenAIServe defaults to "policy"
        self.served_model_name = "policy"
    
    def _maybe_parse_tool_calls(self, texts: list[str]) -> list[dict[str, Any]]:
        """Parse tool calls from generated texts if a parser is configured.
        
        Returns a list aligned with `texts`, each entry a dict (model_dump) or None.
        """
        
        return self.get_deployment_handle().maybe_parse_tool_calls.remote(self.cfg["vllm_cfg"].get("tool_parser", None), texts).result()

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

        # Build prompts for batched request (use vLLM OpenAI extension: prompt_token_ids)
        prompts_token_ids_batch: list[list[int]] = []
        for i in range(batch_size):
            valid_len = int(input_lengths[i].item())
            prompts_token_ids_batch.append(
                input_ids[i, :valid_len].tolist() if valid_len > 0 else []
            )

        # Decide whether we can use a single batched call or must fall back to per-sample
        use_single_call = False
        common_stop_strings: Optional[list[str]] = None
        if not batch_stop_strings or len(batch_stop_strings) == 0:
            use_single_call = True
            common_stop_strings = None
        elif len(batch_stop_strings) >= batch_size and all(
            batch_stop_strings[i] == batch_stop_strings[0] for i in range(batch_size)
        ):
            use_single_call = True
            common_stop_strings = batch_stop_strings[0]

        if use_single_call:
            generated_token_id_lists, generated_logprobs_lists, generated_texts = self._generate_batch(
                prompts_token_ids=prompts_token_ids_batch,
                stop_strings=common_stop_strings,
                greedy=greedy,
            )
        else:
            # Per-sample fallback when stop strings differ across the batch
            generated_token_id_lists = []
            generated_logprobs_lists = []
            generated_texts = []
            for i in range(batch_size):
                stop_strings_i: Optional[list[str]] = None
                if batch_stop_strings and i < len(batch_stop_strings):
                    stop_strings_i = batch_stop_strings[i]
                gen_ids_i, gen_lps_i, gen_text_i = self._generate_batch(
                    prompts_token_ids=[prompts_token_ids_batch[i]],
                    stop_strings=stop_strings_i,
                    greedy=greedy,
                )
                generated_token_id_lists.append(gen_ids_i[0])
                generated_logprobs_lists.append(gen_lps_i[0])
                generated_texts.append(gen_text_i[0])

        max_generated = max((len(ids) for ids in generated_token_id_lists), default=0)

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

            full_output = torch.full((total_length,), pad_id, dtype=torch.long)
            full_output[:seq_len] = input_ids[i][:seq_len].to(dtype=torch.long)
            if len(gen_ids) > 0:
                full_output[seq_len : seq_len + len(gen_ids)] = torch.tensor(gen_ids, dtype=torch.long)

            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            for j, lp in enumerate(gen_lps):
                pos = seq_len + j
                if pos < total_length:
                    full_logprobs[pos] = float(lp)
            logprobs_list.append(full_logprobs)

            generation_lengths.append(len(gen_ids))
            unpadded_sequence_lengths.append(seq_len + len(gen_ids))
                
        tool_calls = self._maybe_parse_tool_calls(generated_texts)

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
                "unpadded_sequence_lengths": torch.tensor(unpadded_sequence_lengths, dtype=torch.long),
                "tool_calls": tool_calls,
            }
        )

    def _generate_batch(
        self,
        *,
        prompts_token_ids: list[list[int]],
        stop_strings: Optional[list[str]],
        greedy: bool,
    ) -> tuple[list[list[int]], list[list[float]], list[str]]:
        max_new_tokens: int = self.cfg["max_new_tokens"]
        temperature: float = 0.0 if greedy else self.cfg["temperature"]
        top_p: float = self.cfg["top_p"]
        top_k: Optional[int] = 1 if greedy else (self.cfg["top_k"] if self.cfg["top_k"] is not None else -1)

        resp = self.client.completions.create(
            model=self.served_model_name,
            prompt=prompts_token_ids,
            max_tokens=self.cfg["max_new_tokens"],
            temperature=temperature,
            top_p=top_p,
            stop=stop_strings if stop_strings else None,
            logprobs=1,
            extra_body={
                "return_tokens_as_token_ids": True,
            }
        )
        
        generated_token_ids_list: list[list[int]] = []
        generated_logprobs_list: list[list[float]] = []
        generated_texts_list: list[str] = []

        for choice in resp.choices:
            generated_logprobs = choice.logprobs.token_logprobs
            
            # If you specify return_tokens_as_token_ids, the tokens are returned as strings like "token_id:1234"
            generated_token_strings = choice.logprobs.tokens
            generated_token_ids = [int(token.split(":")[1]) for token in generated_token_strings]
            
            # Extract text for tool call parsing
            generated_text = choice.text
            
            generated_token_ids_list.append(generated_token_ids)
            generated_logprobs_list.append(generated_logprobs)
            generated_texts_list.append(generated_text)
            
        return generated_token_ids_list, generated_logprobs_list, generated_texts_list

    async def generate_async(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict["GenerationOutputSpec"]], None]:
        raise NotImplementedError("generate_async is not yet supported for vLLM over HTTP")

    def shutdown(self):
        serve.shutdown()

    def get_deployment_handle(self) -> DeploymentHandle:
        return serve.get_deployment_handle("VLLMOpenAIServe", app_name="vllm_http_generation")

    def init_collective(self, ip: str, port: int, world_size: int):
        return [self.get_deployment_handle().admin_init_collective.remote(0, ip, port, world_size)]

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # Wait for the reset to complete across replicas
        self.get_deployment_handle().admin_reset_prefix_cache.remote().result()
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        # Wait for refit prep to complete across replicas.
        self.get_deployment_handle().admin_prepare_refit_info.remote(state_dict_info).result()

    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        raise NotImplementedError("update_weights_from_ipc_handles is not supported for vLLM over HTTP")

    def update_weights_from_collective(self):
        return [self.get_deployment_handle().admin_update_from_collective.remote()]
from typing import Any, Optional, TypedDict, NotRequired
import time

import ray
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import httpx

import verifiers as vf

from openai import AsyncOpenAI
from datasets import Dataset

from rlkit.environments.interfaces import EnvironmentInterface


class VfEnvironmentConfig(TypedDict):
    # The Python import path of the environment (used in vf.load_environment).
    environment_name: str
    # Passed to vf.load_environment as kwargs - make sure this is serializable.
    environment_config: dict[str, Any] | None
    # Timeout settings for OpenAI API client (in seconds).
    # Increase these for large batch sizes on weak GPU machines.
    client_timeout: NotRequired[float]  # Total timeout for requests
    client_connect_timeout: NotRequired[float]  # Connection timeout

@ray.remote(max_restarts=-1, max_task_retries=-1)
class VfEnvironment(EnvironmentInterface):
    """Wraps a verifiers environment in a Ray worker."""
    cfg: VfEnvironmentConfig
    env: vf.MultiTurnEnv

    client: Optional[AsyncOpenAI] = None

    tokenizer: PreTrainedTokenizerBase
    
    def __init__(self, cfg: VfEnvironmentConfig, model_name: str):
        self.cfg = cfg
        
        if "environment_config" not in cfg or cfg["environment_config"] is None:
            env_cfg = {}
        else:
            env_cfg = cfg["environment_config"]
        
        self.env = vf.load_environment(
            cfg["environment_name"],
            **env_cfg,
        )

        # Necessary for verifiers to process results.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # The only default verifiers environment type that isn't compatible with MultiTurnEnv is EnvGroup, which we have a multi-turn replacement for.
        if not isinstance(self.env, vf.MultiTurnEnv):
            raise TypeError("VfEnvironment only supports MultiTurnEnv environments (this includes SingleTurnEnv and ToolEnv but not EnvGroup).")
        
    async def a_generate(
        self,
        inputs: vf.GenerateInputs | Dataset | dict,
        sampling_args: vf.SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> vf.GenerateOutputs:
        if self.client is None:
            # Get timeout settings from config or use defaults.
            # Default timeout is 600 seconds (10 minutes) for large batch processing.
            totalTimeout = self.cfg.get("client_timeout", 600.0)
            connectTimeout = self.cfg.get("client_connect_timeout", 60.0)
            
            # Create httpx client with custom timeouts.
            httpxClient = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=totalTimeout,
                    connect=connectTimeout,
                )
            )
            
            self.client = AsyncOpenAI(
                api_key="n/a",
                base_url="http://127.0.0.1:8000/v1",
                http_client=httpxClient,
            )
        assert isinstance(sampling_args, dict), "sampling_args must be a dictionary."
        
        # Manually call generation and scoring separately to track timing
        if isinstance(inputs, vf.GenerateInputs):
            inputs_dict = inputs.model_dump()
        elif isinstance(inputs, Dataset):
            inputs_dict = {col: inputs[col] for col in inputs.column_names}
        else:
            inputs_dict = inputs
        
        # Prepare inputs
        from copy import deepcopy
        import json
        results_dict = {col: deepcopy(inputs_dict[col]) for col in inputs_dict}
        if "prompt" not in results_dict:
            raise ValueError("prompt column not found in inputs")
        if "answer" not in results_dict:
            results_dict["answer"] = [""] * len(results_dict["prompt"])
        if "task" not in results_dict:
            results_dict["task"] = ["default"] * len(results_dict["prompt"])
        if "info" not in results_dict:
            results_dict["info"] = [{}] * len(results_dict["prompt"])
        
        # Deserialize info if it's a JSON string (for compatibility with dataset storage)
        for i, info in enumerate(results_dict["info"]):
            if isinstance(info, str):
                results_dict["info"][i] = json.loads(info)
        
        import asyncio
        
        # Process rollouts in parallel while tracking individual timings
        num_rollouts = len(results_dict["prompt"])
        
        async def generate_single_rollout(i):
            """Generate a single rollout and return its timing + results"""
            gen_start = time.perf_counter()
            rollout_result = await self.env.run_rollouts(
                prompts=[results_dict["prompt"][i]],
                answers=[results_dict["answer"][i]],
                tasks=[results_dict["task"][i]],
                infos=[results_dict["info"][i]],
                client=self.client,
                model="policy",
                sampling_args=sampling_args,
                max_concurrent=1,  # Each task handles 1 rollout
                **kwargs,
            )
            gen_time = time.perf_counter() - gen_start
            
            completion = rollout_result[0][0]
            state = rollout_result[0][1]
            
            return {
                "index": i,
                "completion": completion,
                "state": state,
                "gen_time": gen_time,
            }
        
        # PHASE 1: Generate all rollouts in parallel
        gen_tasks = [generate_single_rollout(i) for i in range(num_rollouts)]
        gen_results = await asyncio.gather(*gen_tasks)
        
        # Sort by index to maintain order
        gen_results.sort(key=lambda x: x["index"])
        
        # Extract generation results
        completions = [r["completion"] for r in gen_results]
        states = [r["state"] for r in gen_results]
        generation_times = [r["gen_time"] for r in gen_results]
        
        # PHASE 2: Score all rollouts together as a batch (they need to see each other for relative scoring)
        scoring_times = []
        rewards = []
        
        if score_rollouts:
            # Score all rollouts together in one batch call
            score_start = time.perf_counter()
            rollout_scores = await self.env.rubric.score_rollouts(
                prompts=results_dict["prompt"],
                completions=completions,
                answers=results_dict["answer"],
                states=states,
                tasks=results_dict["task"],
                infos=results_dict["info"],
                apply_weights=True,
            )
            score_end = time.perf_counter()
            total_score_time = score_end - score_start
            
            rewards = rollout_scores.reward
            
            # All rollouts scored together - report the total batch time for each
            # (this is the wall-clock time, which equals per-rollout time if they run in parallel)
            scoring_times = [total_score_time] * num_rollouts
        else:
            rewards = [0.0] * num_rollouts
            scoring_times = [0.0] * num_rollouts
        
        results = vf.GenerateOutputs(
            prompt=results_dict["prompt"],
            answer=results_dict["answer"],
            task=results_dict["task"],
            info=results_dict["info"],
            completion=completions,
            state=states,
            reward=rewards,
            metrics={},
        )
        
        # Store per-rollout timing
        results.metrics["generation_time"] = generation_times
        results.metrics["scoring_time"] = scoring_times
        
        return results

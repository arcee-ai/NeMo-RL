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
        
        # Just call the original verifiers a_generate
        results = await self.env.a_generate(
            inputs=inputs,
            client=self.client,
            model="policy",
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            **kwargs,
        )
        
        return results

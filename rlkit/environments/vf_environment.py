from typing import Any, Optional, TypedDict

import ray
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import verifiers as vf

from openai import AsyncOpenAI
from datasets import Dataset

from rlkit.environments.interfaces import EnvironmentInterface


class VfEnvironmentConfig(TypedDict):
    # The Python import path of the environment (used in vf.load_environment).
    environment_name: str
    # Passed to vf.load_environment as kwargs - make sure this is serializable.
    environment_config: dict[str, Any] | None

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

        self.api_ips = []
        self.clients = []
        self.current_client_index = 0
    
    def set_api_ips(self, api_ips: list[str]):
        self.api_ips = api_ips
        self.clients = [AsyncOpenAI(
            api_key="n/a",
            base_url=f"http://{ip}:8000/v1"
        ) for ip in api_ips]
        
    def get_next_client(self) -> AsyncOpenAI:
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client
    
    async def a_generate(
        self,
        inputs: vf.GenerateInputs | Dataset | dict,
        sampling_args: vf.SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> tuple[vf.GenerateOutputs, vf.ProcessedOutputs]:
        assert isinstance(sampling_args, dict), "sampling_args must be a dictionary."
        client = self.get_next_client()
        results = await self.env.a_generate(
            inputs=inputs,
            client=client,
            model="policy",
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            **kwargs,
        )

        return results

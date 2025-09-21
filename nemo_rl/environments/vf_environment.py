from typing import Any, Optional, TypedDict

import ray
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

import verifiers as vf
import vf_exts as vfe

from openai import AsyncOpenAI
import httpx
from datasets import Dataset

class VfEnvironmentMetadata(TypedDict):
    """Persistent state of the environment across steps."""
    answer: str | None
    state: vf.State | None
    task: str | None
    info: vf.Info | None
    metrics: dict[str, float] | None

class VfEnvironmentConfig(TypedDict):
    # The Python import path of the environment (used in vf.load_environment).
    environment_name: str
    # Passed to vf.load_environment as kwargs - make sure this is serializable.
    environment_config: dict[str, Any] | None


def split_prompt_completion(messages: vf.Messages) -> tuple[vf.Messages, vf.Messages]:
    """Split a list of messages into a prompt and completion."""
    last_user_index = 0
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_index = i
    return messages[:last_user_index+1], messages[last_user_index+1:]

@ray.remote(max_restarts=-1, max_task_retries=-1)
class VfEnvironment(EnvironmentInterface[VfEnvironmentMetadata]):
    """Wraps a verifiers environment into a NeMo-RL one."""
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
        
    async def step(self, message_log_batch: list[LLMMessageLogType], metadata: list[VfEnvironmentMetadata]) -> EnvironmentReturn[VfEnvironmentMetadata]:
        raise NotImplementedError("VfEnvironment only supports verifiers rollouts.")
        
    def global_post_process_and_metrics(self, batch: BatchedDataDict[Any]) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        raise NotImplementedError("VfEnvironment only supports verifiers rollouts.")
    
    async def a_generate(
        self,
        inputs: vf.GenerateInputs | Dataset | dict,
        sampling_args: vf.SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> tuple[vf.GenerateOutputs, vf.ProcessedOutputs]:
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key="n/a",
                base_url="http://127.0.0.1:8000/v1",
                http_client=httpx.AsyncClient(
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=28000,
                        max_keepalive_connections=28000,
                    ),
                    timeout=600.0,
                ),
            )
        assert isinstance(sampling_args, dict), "sampling_args must be a dictionary."
        results = await self.env.a_generate(
            inputs=inputs,
            client=self.client,
            model="policy",
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            **kwargs,
        )

        # Process the environment results for things like tokenization and logprobs.
        return results, self.env.process_env_results_vllm(
            prompts=results.prompt,
            completions=results.completion,
            states=results.state,
            rewards=results.reward,
            processing_class=self.tokenizer,
        )
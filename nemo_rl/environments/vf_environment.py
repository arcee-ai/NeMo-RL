from typing import Any, TypedDict
import logging

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

import verifiers as vf

from nemo_rl.models.generation.interfaces import GenerationInterface

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
    
    
@ray.remote(max_restarts=-1, max_task_retries=-1)
class VfEnvironment(EnvironmentInterface[VfEnvironmentMetadata]):
    """Wraps a verifiers environment into a NeMo-RL one."""
    cfg: VfEnvironmentConfig
    env: vf.MultiTurnEnv
    generation: GenerationInterface | None
    
    def __init__(self, cfg: VfEnvironmentConfig):
        self.cfg = cfg
        
        if "environment_config" not in cfg or cfg["environment_config"] is None:
            env_cfg = {}
        else:
            env_cfg = cfg["environment_config"]
        
        self.env = vf.load_environment(
            cfg["environment_name"],
            **env_cfg,
        )
        
        # The only default verifiers environment type that isn't compatible with MultiTurnEnv is EnvGroup.
        # TODO: Work out a replacement for EnvGroup with proper MultiTurnEnv support.
        if not isinstance(self.env, vf.MultiTurnEnv):
            raise TypeError("VfEnvironment only supports MultiTurnEnv environments (this includes SingleTurnEnv and ToolEnv but not EnvGroup).")
        
        # Optional handle to a generation controller (e.g., VllmGeneration)
        self.generation: Any = None

    def set_generation_handle(self, generation: GenerationInterface) -> None:
        """Store a handle to a generation interface (e.g., VllmGeneration)."""
        self.generation = generation

    def query_vllm(self, prompts: list[str], greedy: bool = False) -> list[str]:
        """Query the vLLM generation backend from within the environment worker."""
        assert self.generation is not None, "vLLM generation handle not set"
        inputs = BatchedDataDict({"prompts": prompts})
        outputs = self.generation.generate_text(inputs, greedy=greedy)
        return outputs["texts"]
        
        
    async def step(self, message_log_batch: list[LLMMessageLogType], metadata: list[VfEnvironmentMetadata]) -> EnvironmentReturn[VfEnvironmentMetadata]:
        observations = []
        next_metadata = []
        next_stop_strings = []
        rewards = []  # Gets converted to tensor at the end.
        terminated = []  # Also gets converted to tensor.
        answers = []
        
        print(self.query_vllm([{"role": "user", "content": "Hello, how are you?"}]))
        
        for messages, meta in zip(message_log_batch, metadata):
            # If state is None, initialize a new state mimicking how the verifiers rollout system would.
            if meta.get("state", None) is None:
                # Find last user message index.
                last_user_index = 0
                for i, m in enumerate(messages):
                    if m.get("role") == "user":
                        last_user_index = i
                
                meta["state"] = self.env.setup_state({
                    "prompt": messages[:last_user_index+1],
                    "completion": messages[last_user_index+1:],
                    "answer": meta.get("answer", None),
                    "task": meta.get("task", "default"),
                    "info": meta.get("info", None),
                    "responses": messages[last_user_index+1:],
                    "turn": sum(1 for m in messages if m.get("role") == "assistant"),
                })
            
            # Step verifiers environment.
            responses, new_state = self.env.env_response(messages, meta["state"])
            meta["state"].update(new_state)
            observations.append(responses)
            
            if len(responses) == 0:
                observations.append({"role": "environment", "content": "generic termination feedback"})
            
            if self.env.is_completed(messages, meta["state"]) or meta["state"]["turn"] >= self.env.max_turns:
                # Rollout marked complete - calculate rewards and finalize.
                
                # TODO: Make some affordance for Kimi-style group-relative judgement here.
                # We're using standard verifiers-style reward functions, which evaluate in a vacuum.
                results: vf.RolloutScore = await self.env.rubric.score_rollout(
                    meta["state"]["prompt"],
                    meta["state"]["completion"],
                    meta.get("answer", None),
                    meta["state"],
                    meta.get("task", None),
                    meta.get("info", None),
                )
                
                # This gets integrated into the rollout metrics.
                meta["metrics"] = results.metrics
                
                # There isn't another rollout after this one, so the model doesn't actually see this.
                next_metadata.append(meta)
                next_stop_strings.append(None)
                rewards.append(results.reward)
                terminated.append(True)
            else:
                # Largely placeholders to indicate a follow-up step is required.
                next_metadata.append(meta)
                # TODO: Add support for this kind of stop-string-based rollout interruption.
                next_stop_strings.append(None)
                rewards.append(0)
                terminated.append(False)
            
            # Use verifiers answer field if present.
            answers.append(meta.get("answer", None))
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards).cpu(),
            terminateds=torch.tensor(terminated).cpu(),
            answers=answers,
        )
        
    def global_post_process_and_metrics(self, batch: BatchedDataDict[Any]) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        # TODO: Find a way to smuggle the metrics from the rubric grading results into this function.
        return batch, {
            # Basic metrics from the info we have.
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
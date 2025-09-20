from typing import Any, Optional, TypedDict
import logging

import ray
import torch

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

import asyncio

async def run_rollouts(
    self,
    client: AsyncOpenAI,
    model: str,
    prompts: list[vf.Messages],
    answers: list[str],
    tasks: list[str],
    infos: list[vf.Info],
    sampling_args: vf.SamplingArgs | None = None,
    max_concurrent: int = -1,
    **kwargs,
) -> list[tuple[vf.Messages, vf.State]]:
    """
    Batched multi-turn rollouts: issue one batched chat.completions call per turn
    and advance all conversations using a simple queue until completion.
    """
    assert self.message_type == "chat", "Batched run_rollouts only supports chat format"

    n = len(prompts)
    results: list[tuple[vf.Messages, vf.State]] = [([], {}) for _ in range(n)]

    states: list[vf.State] = []
    rollouts: list[vf.Messages] = []
    completions: list[vf.Messages] = []
    done: list[bool] = [False] * n

    for i in range(n):
        info = infos[i] or {}
        prompt_i = prompts[i]
        answer_i = answers[i] if answers is not None and i < len(answers) else ""
        task_i = tasks[i] if tasks is not None and i < len(tasks) else "default"

        state: vf.State = {
            "prompt": prompt_i,
            "completion": [],
            "answer": answer_i,
            "task": task_i,
            "info": info,
            "responses": [],
            "turn": 0,
        }
        state = await self.setup_state(state, **kwargs)
        rollout_i: vf.Messages = list(prompt_i)  # type: ignore
        completion_i: vf.Messages = []

        states.append(state)
        rollouts.append(rollout_i)
        completions.append(completion_i)

    # Normalize sampling args
    sa = dict(sampling_args or {})
    if "max_tokens" in sa and self.message_type == "chat":
        if sa.get("max_tokens", None) is None:
            sa.pop("max_tokens", None)
        else:
            sa["max_completion_tokens"] = sa.pop("max_tokens")
    if sa.get("max_completion_tokens", None) is None:
        sa.pop("max_completion_tokens", None)
    clean_sampling_args = {k: v for k, v in sa.items() if v is not None}

    # Ensure tools are consistent across the batch if provided
    tool_sets = []
    for info in infos:
        ts = info.get("oai_tools", None) if info is not None else None
        tool_sets.append(ts)
    unique_tool_keys = []
    for ts in tool_sets:
        if ts not in unique_tool_keys:
            unique_tool_keys.append(ts)
    if len(unique_tool_keys) > 1:
        raise ValueError("All rollouts must share the same oai_tools for batched chat completions")
    oai_tools = unique_tool_keys[0] if unique_tool_keys else None

    # Queue: advance all active conversations turn-by-turn
    active_indices = [i for i in range(n)]
    while len(active_indices) > 0:
        batch_messages: list[vf.Messages] = [rollouts[i] for i in active_indices]

        # Single batched request for all active conversations
        if oai_tools:
            resp = await client.chat.completions.create(
                model=model,
                messages=batch_messages,  # type: ignore
                tools=oai_tools,
                **clean_sampling_args,
            )
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=batch_messages,  # type: ignore
                **clean_sampling_args,
            )

        assert hasattr(resp, "choices") and len(resp.choices) >= len(active_indices), "Batched response mismatch"

        # Map responses back to active indices
        new_active: list[int] = []
        for j, idx in enumerate(active_indices):
            choice = resp.choices[j]
            response_text = (choice.message.content or "")
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response_text}
            if getattr(choice.message, "tool_calls", None):
                assistant_msg["tool_calls"] = choice.message.tool_calls  # type: ignore

            rollouts[idx].append(assistant_msg)  # type: ignore
            completions[idx].append(assistant_msg)  # type: ignore
            states[idx]["responses"].append(resp)
            states[idx]["turn"] += 1

            is_done = await self.is_completed(rollouts[idx], states[idx], **kwargs)
            if (self.max_turns > 0 and states[idx]["turn"] >= self.max_turns) or is_done:
                done[idx] = True
                continue

            env_msgs, new_state = await self.env_response(rollouts[idx], states[idx], **kwargs)
            states[idx].update(new_state)
            if isinstance(env_msgs, list):
                rollouts[idx] += env_msgs  # type: ignore
                completions[idx] += env_msgs  # type: ignore
            else:
                rollouts[idx] = (rollouts[idx] or [])  # type: ignore
                completions[idx] = (completions[idx] or [])  # type: ignore

            new_active.append(idx)

        active_indices = new_active

    for i in range(n):
        results[i] = (completions[i], states[i])

    return results

@ray.remote(max_restarts=-1, max_task_retries=-1)
class VfEnvironment(EnvironmentInterface[VfEnvironmentMetadata]):
    """Wraps a verifiers environment into a NeMo-RL one."""
    cfg: VfEnvironmentConfig
    env: vf.MultiTurnEnv

    client: Optional[AsyncOpenAI] = None
    
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

        from types import MethodType

        # TODO: This is truly awful.
        self.env.run_rollouts = MethodType(run_rollouts, self.env)
        
        # The only default verifiers environment type that isn't compatible with MultiTurnEnv is EnvGroup, which we have a multi-turn replacement for.
        if not isinstance(self.env, vf.MultiTurnEnv):
            raise TypeError("VfEnvironment only supports MultiTurnEnv environments (this includes SingleTurnEnv and ToolEnv but not EnvGroup).")
        
    async def step(self, message_log_batch: list[LLMMessageLogType], metadata: list[VfEnvironmentMetadata]) -> EnvironmentReturn[VfEnvironmentMetadata]:
        observations = []
        next_metadata = []
        next_stop_strings = []
        rewards = []  # Gets converted to tensor at the end.
        terminated = []  # Also gets converted to tensor.
        answers = []
        
        # Gather GRPO groups and populate initial state field.
        group_messages = {}
        group_metas = {}
        for messages, meta in zip(message_log_batch, metadata):
            # If state is None, initialize a new state mimicking how the verifiers rollout system would.
            if meta.get("state", None) is None:
                prompt, completion = split_prompt_completion(messages)
                meta["state"] = await self.env.setup_state({
                    "prompt": prompt,
                    "completion": completion,
                    "answer": meta.get("answer", None),
                    "task": meta.get("task", None),
                    "info": meta.get("info", None),
                    "responses": completion,
                    "turn": sum(1 for m in messages if m.get("role") == "assistant"),
                })
            
            if meta.get("_grpo_gid", None) is None:
                raise ValueError("GRPO group not specified for rollout.", messages, meta)
            
            grpo_gid: int = meta["_grpo_gid"]
            if grpo_gid not in group_messages:
                group_messages[grpo_gid] = []
                group_metas[grpo_gid] = []
            group_messages[grpo_gid].append(messages)
            group_metas[grpo_gid].append(meta)
        
        grouped_rewards: dict[int, list[float]] = {}
        grouped_metrics: dict[int, dict[str, list[float]]] = {}
        
        if isinstance(self.env.rubric, vfe.GroupedRubric):
            # Validate all groups are the same size.
            normal_length = None
            for grpo_gid, messages in group_messages.items():
                if normal_length is None:
                    normal_length = len(messages)
                elif len(messages) != normal_length:
                    # TODO: Figure out how to support this with async rollout generation.
                    raise ValueError(f"GRPO group {grpo_gid} has different number of messages and metas. Please ensure you are not using the async engine, as this is currently incompatible with grouped rubrics.")
            
            # Score groups
            for grpo_gid in group_messages.keys():
                all_messages = group_messages[grpo_gid]
                metas = group_metas[grpo_gid]
                assert len(set(meta.get("task", None) for meta in metas)) <= 1, "All rollouts in a GRPO group should have the same task."
                
                split = [split_prompt_completion(m) for m in all_messages]
                prompts = [x[0] for x in split]
                completions = [x[1] for x in split]
                
                states = [x["state"] for x in metas]
                
                results: vf.RolloutScores = await self.env.rubric.score_rollouts_grouped(
                    prompts,
                    completions,
                    metas[0].get("answer", None),
                    states,
                    metas[0].get("task", None),
                    metas[0].get("info", None),
                )
                grouped_rewards[grpo_gid] = results.reward
                grouped_metrics[grpo_gid] = results.metrics
                
                # Tag message with their group index.
                for i, meta in enumerate(metas):
                    meta["_grpo_group_idx"] = i
        
        for messages, meta in zip(message_log_batch, metadata):
            # Step verifiers environment.
            responses, new_state = await self.env.env_response(messages, meta["state"])
            meta["state"].update(new_state)
            observations.append(responses)
            
            if len(responses) == 0:
                observations.append({"role": "environment", "content": "generic termination feedback"})
            
            env_is_completed = await self.env.is_completed(messages, meta["state"])
            if env_is_completed or meta["state"]["turn"] >= self.env.max_turns:
                # Rollout marked complete - calculate rewards and finalize.
                
                if meta.get("_grpo_gid", None) in grouped_rewards:
                    rewards.append(grouped_rewards[meta["_grpo_gid"]][meta["_grpo_group_idx"]])
                    
                    meta["metrics"] = grouped_metrics[meta["_grpo_gid"]]
                else:
                    results: vf.RolloutScore = await self.env.rubric.score_rollout(
                        meta["state"]["prompt"],
                        meta["state"]["completion"],
                        meta.get("answer", None),
                        meta["state"],
                        meta.get("task", None),
                        meta.get("info", None),
                    )
                    
                    rewards.append(results.reward)
                
                    # This later gets integrated into the rollout metrics.
                    meta["metrics"] = results.metrics
                
                # There isn't another rollout after this one, so the model doesn't actually see this.
                next_metadata.append(meta)
                next_stop_strings.append(None)
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
    
    async def a_generate(
        self,
        inputs: vf.GenerateInputs | Dataset | dict,
        sampling_args: vf.SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> vf.GenerateOutputs:
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
        return await self.env.a_generate(
            inputs=inputs,
            client=self.client,
            model="policy",
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            **kwargs,
        )
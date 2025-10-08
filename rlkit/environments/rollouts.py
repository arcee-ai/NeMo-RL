from typing import Any

from openai.types.chat import ChatCompletion
import torch
import ray
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.data.interfaces import DatumSpec
from rlkit.models.generation.vllm_http.vllm_http_generation import VllmHttpGeneration
from rlkit.environments.interfaces import EnvironmentInterface

import verifiers as vf

def split_rollouts_by_group(
    prompts: list[vf.Messages],
    answers: list[str],
    infos: list[vf.Info],
    tasks: list[str],
    grpo_gids: list[int],
) -> dict[int, list[tuple[vf.Messages, str, vf.Info, str, int]]]:
    """
    Split rollouts by GRPO group.

    Args:
        prompts: List of prompts.
        answers: List of answers.
        infos: List of infos.
        tasks: List of tasks.
        grpo_gids: List of GRPO group IDs.

    Returns:
        Dictionary of GRPO group IDs to lists of tuples of (prompt, answer, info, task, grpo_gid, orig_idx).
    """
    results_by_group = {}
    for i, rollout in enumerate(zip(prompts, answers, infos, tasks, grpo_gids)):
        if rollout[-1] not in results_by_group:
            results_by_group[rollout[-1]] = []
        results_by_group[rollout[-1]].append(rollout + (i,))
    return results_by_group


def build_rollouts_log(
    message_logs: list[list[dict[str, Any]]],
    grpo_group_ids: list[int],
    total_rewards: list[float],
    extra_env_infos: list[dict[str, Any]],
    sample_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construct rich per-sample rollout logs for metrics dashboards."""

    rollout_log: list[dict[str, Any]] = []
    for i in range(len(message_logs)):
        metrics = sample_metrics[i]
        rollout_log.append(
            {
                "messages": message_logs[i],
                "grpo_group_id": grpo_group_ids[i],
                "total_reward": float(total_rewards[i]),
                "terminated": bool(metrics.get("terminated", False)),
                "truncated": bool(metrics.get("truncated", False)),
                "total_tokens": int(metrics.get("total_tokens", 0)),
                "assistant_tokens": int(metrics.get("assistant_tokens", 0)),
                "env_tokens": int(metrics.get("env_tokens", 0)),
                "env_metrics": extra_env_infos[i].get("metrics", {}),
            }
        )

    return rollout_log

def run_vf_rollouts(
    policy_generation: VllmHttpGeneration,
    input_batch: BatchedDataDict[DatumSpec],
    vf_semaphore: int | None,
    max_seq_len: int,
    max_new_tokens: int | None,
    env: EnvironmentInterface,
    grpo_gids: list[int],
    greedy: bool = False,
):
    assert isinstance(policy_generation, VllmHttpGeneration), "Verifiers environments require a vLLM client."
    
    prompt = input_batch["prompt"]
    info = input_batch["info"]
    answer = input_batch["answer"]
    task = input_batch["task"]

    by_group = split_rollouts_by_group(prompt, answer, info, task, grpo_gids)

    # Convert input batch to verifiers input format
    verifiers_input_batches = []

    for group in by_group.values():
        group_prompts = []
        group_answers = []
        group_infos = []
        group_tasks = []
        for rollout in group:
            group_prompts.append(rollout[0])
            group_answers.append(rollout[1])
            group_infos.append(rollout[2])
            group_tasks.append(rollout[3])
        
        verifiers_input_batches.append({
            "prompt": group_prompts,
            "answer": group_answers,
            "info": group_infos,
            "task": group_tasks,
        })
    
    sampling_args = {
        "max_tokens": max_new_tokens,
        "top_p": policy_generation.cfg.get("top_p", None),
        "temperature": policy_generation.cfg.get("temperature", None),
        "stop_strings": policy_generation.cfg.get("stop_strings", None),
        "logprobs": 1,
        "extra_body": {
            "return_tokens_as_token_ids": True,
            "top_k": policy_generation.cfg.get("top_k", None),
        }
    }

    if greedy:
        sampling_args["temperature"] = 0.0

    if vf_semaphore is not None:
        batch_semaphore = max(1, vf_semaphore // len(verifiers_input_batches))
    else:
        batch_semaphore = -1

    refs = [
        env.a_generate.remote(
            inputs=verifiers_input_batch,
            sampling_args=sampling_args,
            max_concurrent=batch_semaphore,
        )
        for verifiers_input_batch in verifiers_input_batches
    ]
    generate_results = ray.get(refs)

    current_batch = input_batch.copy()
    current_batch["reward"] = [0 for _ in current_batch["prompt"]]

    env_metrics_sums = {}
    env_metrics_counts = {}

    sample_truncated = [False for _ in current_batch["prompt"]]
    
    current_batch["completion"] = [[] for _ in current_batch["prompt"]]

    # Convert completion to RLKit message log format.
    for g_i, rollouts in enumerate(generate_results):
        # type hints for convenience
        rollouts: vf.GenerateOutputs
        
        for i, completion in enumerate(rollouts.completion):
            assert isinstance(completion, list), "RLKit currently only supports chat completions."
            
            if "responses" not in rollouts.state[i]:
                raise ValueError("No chat completions API responses found in rollouts.state.")
            
            responses: list[ChatCompletion] = rollouts.state[i]["responses"]
            responses_idx = 0

            for key, value in rollouts.metrics.items():
                if isinstance(value, (int, float)):
                    if key not in env_metrics_sums:
                        env_metrics_sums[key] = 0.0
                        env_metrics_counts[key] = 0
                    env_metrics_sums[key] += value
                    env_metrics_counts[key] += 1

            log = []
            orig_idx = list(by_group.values())[g_i][i][-1]
            for msg in completion:
                if msg["role"] == "assistant":
                    result = responses[responses_idx]
                    responses_idx += 1
                    
                    logprobs_obj = result.choices[0].logprobs.content
                    
                    token_ids = []
                    generation_logprobs = []
                    for logprob_obj in logprobs_obj:
                        token_ids.append(int(logprob_obj.token.split(":")[-1]))
                        generation_logprobs.append(logprob_obj.logprob)
                    
                    # We patch vLLM to return the full canonical prompt alongside our completion, so we should see at least one -9999 in the completion logprobs.
                    assert -9999 in generation_logprobs, "vLLM full-sequence monkey-patch appears to have failed, no prompt tokens in completion logprobs"

                    log.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "tool_calls": msg.get("tool_calls", []) or [],
                        "token_ids": torch.tensor(token_ids),
                        "generation_logprobs": torch.tensor(generation_logprobs),
                    })
                else:
                    log.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "tool_calls": msg.get("tool_calls", []) or []
                    })

            current_batch["completion"][orig_idx].extend(log)
            current_batch["reward"][orig_idx] = rollouts.reward[i]

    current_batch["reward"] = torch.tensor(current_batch["reward"])

    env_metrics_means = {k: v / env_metrics_counts[k] for k, v in env_metrics_sums.items()}

    # Compute per-sample metrics to populate rollout logs
    batch_size = len(current_batch["prompt"])

    sample_assistant_tokens: list[int] = []
    sample_total_tokens: list[int] = []

    for completion in current_batch["completion"]:
        assistant_tokens = 0
        for msg in completion:
            if msg.get("role") == "assistant":
                token_ids = msg["token_ids"].tolist()
                
                assistant_tokens += len([x for i, x in enumerate(token_ids) if msg["generation_logprobs"][i] != -9999])
        sample_total_tokens.append(completion[-1]["token_ids"].numel())
        sample_assistant_tokens.append(assistant_tokens)

    sync_sample_metrics = [
        {
            "terminated": False,
            "truncated": sample_truncated[i],
            "assistant_tokens": sample_assistant_tokens[i],
            "total_tokens": sample_total_tokens[i]
        }
        for i in range(batch_size)
    ]

    # Aggregate rollout metrics to mirror legacy rollout logging keys
    denom = max(batch_size, 1)
    rollout_metrics = {
        "total_turns": batch_size,
        "avg_turns_per_sample": 1.0,
        "max_turns_per_sample": 1,
        "natural_termination_rate": 0.0,
        "truncation_rate": 0.0,
        "max_turns_reached_rate": 0.0,
        "mean_total_tokens_per_sample": float(sum(sample_total_tokens) / denom),
        "mean_gen_tokens_per_sample": float(sum(sample_assistant_tokens) / denom),
        "mean_env_tokens_per_sample": 0.0,
        "env_metrics": env_metrics_means,
    }

    rollout_metrics["rollouts/text"] = build_rollouts_log(
        message_logs=[prompt + completion for prompt, completion in zip(current_batch["prompt"], current_batch["completion"])],
        grpo_group_ids=current_batch["idx"],
        total_rewards=[current_batch["reward"][i].item() for i in range(len(current_batch["reward"]))],
        extra_env_infos=[{"metrics": env_metrics_means} for _ in range(len(current_batch["prompt"]))],
        sample_metrics=sync_sample_metrics,
    )

    return current_batch, rollout_metrics

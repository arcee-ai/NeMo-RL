import torch
import ray
from nemo_rl.models.generation.vllm_http.vllm_http_generation import VllmHttpGeneration
from nemo_rl.experience.rollouts import BatchedDataDict, DatumSpec, TokenizerType, LLMMessageLogType
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import build_rollouts_log

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

def run_vf_rollouts(
    policy_generation: VllmHttpGeneration,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    max_new_tokens: int,
    task_to_env: dict[str, EnvironmentInterface],
    grpo_gids: list[int],
    greedy: bool = False,
):
    for env in task_to_env.values():
        assert hasattr(env, "a_generate"), "Verifiers environments require a VfEnvironment."
    
    assert len(task_to_env) == 1, "Verifiers environments are not supported with multiple NeMo-RL tasks. Use vf_exts.MultiTurnEnvGroup instead."
    
    env = list(task_to_env.values())[0]

    assert isinstance(policy_generation, VllmHttpGeneration), "Verifiers environments require a vLLM client."

    vf_msg_log: list[vf.Messages] = []

    for messages in input_batch["message_log"]:
        log = []
        for message in messages:
            entry = {
                "role": message["role"],
                "content": message["content"],
            }
            # Only include tool_calls for assistant messages when present and non-empty
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and len(message["tool_calls"]) > 0
            ):
                entry["tool_calls"] = message["tool_calls"]
            log.append(entry)
        
        vf_msg_log.append(log)
    
    info = [x.get("info", {}) for x in input_batch["extra_env_info"]]
    answer = [x.get("answer", "") for x in input_batch["extra_env_info"]]
    task = [x.get("task", "vf_placeholder") for x in input_batch["extra_env_info"]]

    by_group = split_rollouts_by_group(vf_msg_log, answer, info, task, grpo_gids)

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

    refs = [
        env.a_generate.remote(
            inputs=verifiers_input_batch,
            sampling_args=sampling_args,
            max_concurrent=256,
        )
        for verifiers_input_batch in verifiers_input_batches
    ]
    generate_results = ray.get(refs)

    current_batch = input_batch.copy()
    current_batch["total_reward"] = [0 for _ in current_batch["message_log"]]

    env_metrics = [None for _ in current_batch["message_log"]]

    # Convert completion to NeMo-RL message log format.
    for g_i, (rollouts, processed_outputs) in enumerate(generate_results):
        # type hints for convenience
        rollouts: vf.GenerateOutputs
        processed_outputs: vf.ProcessedOutputs

        for i, completion in enumerate(rollouts.completion):
            assert isinstance(completion, list), "NeMo-RL currently only supports chat completions."

            log = []
            for msg in completion:
                completion_ids = processed_outputs.completion_ids[i]

                log.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "tool_calls": msg.get("tool_calls", []) or [],
                    "token_ids": torch.tensor(completion_ids),
                    "generation_logprobs": torch.tensor(processed_outputs.completion_logprobs[i]),
                })
            
            orig_idx = by_group[g_i][i][-1]

            current_batch["message_log"][orig_idx].extend(log)
            current_batch["total_reward"][orig_idx] = rollouts.reward[i]
            env_metrics[orig_idx] = rollouts.metrics[i]

    # Compute per-sample metrics similar to rollouts.run_multi_turn_rollout
    batch_size = len(current_batch["message_log"])

    sample_assistant_tokens: list[int] = []
    sample_total_tokens: list[int] = []

    for log in current_batch["message_log"]:
        assistant_tokens = 0
        for msg in log:
            if msg.get("role") == "assistant":
                token_ids = msg.get("token_ids", [])
                # token_ids may be a list[int] or torch.Tensor; handle both
                if isinstance(token_ids, torch.Tensor):
                    assistant_tokens += int(token_ids.numel())
                else:
                    assistant_tokens += len(token_ids)
        sample_assistant_tokens.append(assistant_tokens)
        sample_total_tokens.append(assistant_tokens)  # No environment messages added in this path

    sync_sample_metrics = [
        {
            "terminated": False,
            "truncated": False,
            "total_tokens": sample_total_tokens[i],
            "assistant_tokens": sample_assistant_tokens[i],
            "env_tokens": 0,
            env_metrics: env_metrics[i],
        }
        for i in range(batch_size)
    ]

    # Aggregate rollout metrics to mirror keys from run_multi_turn_rollout
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
        "mean_env_tokens_per_sample": 0.0
    }

    rollout_metrics["rollouts/text"] = build_rollouts_log(
        message_logs=current_batch["message_log"],
        grpo_group_ids=current_batch["idx"],
        total_rewards=[current_batch["total_reward"][i].item() for i in range(len(current_batch["message_log"]))],
        extra_env_infos=current_batch["extra_env_info"],
        sample_metrics=sync_sample_metrics,
    )

    return current_batch, rollout_metrics
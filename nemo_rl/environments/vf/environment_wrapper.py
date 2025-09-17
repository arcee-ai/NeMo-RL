from copy import deepcopy
from re import L

import ray
from nemo_rl.models.generation.vllm_http.vllm_http_generation import VllmHttpGeneration
from nemo_rl.experience.rollouts import BatchedDataDict, DatumSpec, TokenizerType, LLMMessageLogType
from nemo_rl.environments.vf_environment import VfEnvironment
from nemo_rl.environments.interfaces import EnvironmentInterface

import verifiers as vf
from openai import AsyncOpenAI

def run_vf_rollouts(
    policy_generation: VllmHttpGeneration,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = -1,
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
            log.append({
                "role": message["role"],
                "content": message["content"],
                "tool_calls": message.get("tool_calls", None)
            })
        
        vf_msg_log.append(log)
    
    info = [x.get("info", None) for x in input_batch["extra_env_info"]]
    answer = [x.get("answer", None) for x in input_batch["extra_env_info"]]
    task = [x.get("task", "vf_placeholder") for x in input_batch["extra_env_info"]]

    # Convert input batch to verifiers input format
    verifiers_input_batch = vf.GenerateInputs(
        prompt=vf_msg_log,
        answer=answer,
        info=info,
        task=task
    )

    sampling_args = {}

    if greedy:
        sampling_args["temperature"] = 0.0

    rollout = ray.get(env.a_generate.remote(
        inputs=verifiers_input_batch,
        sampling_args=sampling_args
    ))

    current_batch = input_batch.copy()
    current_batch["total_reward"] = rollout.reward

    # Convert completion to NeMo-RL message log format, and re-tokenize the prompt to avoid including generation prompts.
    new_msg_logs: list[LLMMessageLogType] = []
    for prompt, completion in zip(rollout.prompt, rollout.completion):
        assert isinstance(completion, list), "NeMo-RL currently only supports chat completions."

        # Use tokenizer args specified from the data processor
        tokenizer_kwargs = input_batch["message_log"][0]["tokenizer_kwargs"]

        log = []
        for i, msg in enumerate(prompt + completion):
            add_tools = (
                i == 0
            )

            # Prevent the template from adding a tool calling prompt to literally every message.
            chat_template_kwargs = deepcopy(tokenizer_kwargs["chat_template_kwargs"])
            if not add_tools:
                chat_template_kwargs["tools"] = None

            chat_text = tokenizer.apply_chat_template(
                [msg],
                # No generation prompt, since we're finished.
                add_generation_prompt=False,
                **chat_template_kwargs
            )
            token_ids = tokenizer(
                chat_text,
                **tokenizer_kwargs["tokenize_kwargs"]
            )["input_ids"][0]

            # TODO: Add generation logprobs
            log.append({
                "role": msg["role"],
                "content": msg["content"],
                "tool_calls": msg["tool_calls"],
                "token_ids": token_ids
            })
        new_msg_logs.append(log)

    current_batch["message_log"] = new_msg_logs
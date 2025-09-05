#!/usr/bin/env python3
"""Quick-and-dirty evaluation script using OpenRouter.

Public objects:
- main

External dependencies:
- verifiers (pip)
- openai (via verifiers)

Usage example:
    # Evaluate vf_basicqa with OpenRouter openai/gpt-5
    python eval_openrouter.py vf_basicqa -n 3 -r 1 \
        -b https://openrouter.ai/api/v1 -k OPENROUTER_API_KEY -m openai/gpt-5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset
from openai import OpenAI

import verifiers as vf
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


class _RubricAdapter:
    """Adapter to bridge grouped rubric APIs to `verifiers` expectations.

    This wraps a rubric that implements `score_rollouts_grouped(...)` with a
    `score_rollouts(...)` method accepting plural list arguments as used by
    `verifiers.envs.Environment`.

    Args:
        base (object): The underlying rubric instance (e.g., GroupedRubric).
    """

    def __init__(self, base: object) -> None:
        self.base = base

    def __getattr__(self, name: str):  # passthrough for other attributes
        return getattr(self.base, name)

    async def score_rollouts(
        self,
        prompts,
        completions,
        answers,
        states,
        tasks,
        infos,
        apply_weights: bool = True,
        **kwargs,
    ):
        # Many grouped rubric implementations expect a single shared answer/task/info
        # for the whole group. We conservatively pass the first element and require
        # lists be non-empty; fall back to sensible defaults otherwise.
        shared_answer = answers[0] if isinstance(answers, list) and answers else ""
        shared_task = tasks[0] if isinstance(tasks, list) and tasks else "default"
        shared_info = infos[0] if isinstance(infos, list) and infos else {}

        # Forward to the grouped API if present; otherwise, raise.
        if hasattr(self.base, "score_rollouts_grouped"):
            return await self.base.score_rollouts_grouped(
                prompts=prompts,
                completions=completions,
                answer=shared_answer,
                states=states,
                task=shared_task,
                info=shared_info,
                **kwargs,
            )
        raise AttributeError("Underlying rubric does not support grouped scoring.")


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent_requests: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    verbose: bool,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
) -> None:
    client = OpenAI(api_key=os.getenv(api_key_var, "EMPTY"), base_url=api_base_url)
    vf_env = vf.load_environment(env_id=env, **env_args)

    # Wrap rubric to support grouped scoring API expected by verifiers.
    if hasattr(vf_env, "rubric") and hasattr(vf_env.rubric, "score_rollouts_grouped"):
        vf_env.rubric = _RubricAdapter(vf_env.rubric)

    # Merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if sampling_args is not None:
        merged_sampling_args.update(sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = temperature

    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=merged_sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent_requests,
    )

    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")

    print("--- Example ---")
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]
    vf.print_prompt_completions_sample(
        printable_prompts, printable_completions, results.reward, step=0
    )

    print("--- All ---")
    print("Rewards:")
    print(
        f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}"
    )

    n = num_examples
    r = rollouts_per_example
    if n < 0:
        n = len(results.reward) // r
    for i in range(r):
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)

    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)

    if save_dataset or save_to_hf_hub:
        ids = [i // rollouts_per_example for i in range(num_examples * rollouts_per_example)]
        rewards = results.reward
        tasks = results.task
        data_dict = {
            "id": ids,
            "prompts": [sanitize_tool_calls(p) for p in printable_prompts],
            "completions": [sanitize_tool_calls(c) for c in printable_completions],
            "task": tasks,
        }
        if results.info[0] != {}:
            data_dict["info"] = results.info
        if results.answer[0] != "":
            data_dict["answer"] = results.answer
        data_dict["reward"] = rewards
        for k in results.metrics:
            v = results.metrics[k]
            data_dict[k] = v

        dataset = Dataset.from_dict(data_dict)
        metadata = {
            "env": env,
            "model": model,
            "num_examples": num_examples,
            "rollouts_per_example": rollouts_per_example,
            "sampling_args": merged_sampling_args,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "avg_reward": sum(results.reward) / len(results.reward),
        }
        for k in results.metrics:
            metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        module_name = env.replace("-", "_")
        local_env_dir = Path("./examples/vf-envs") / module_name
        if local_env_dir.exists():
            results_path = local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
        else:
            results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
        results_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(results_path / "results.jsonl")
        with open(results_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        print(f"Saved dataset to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, default="vf_basicqa", help="Environment module name")
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment args as JSON (e.g., "{\"key\": \"value\", \"num\": 42}")',
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-5-chat",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for API",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=1,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=5,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        "-c",
        type=int,
        default=20,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (unset to use model default)",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help=(
            "Sampling arguments as JSON object. Keys here override --max-tokens/--temperature. "
            'Example: "{\"enable_thinking\": false, \"max_tokens\": 256}"'
        ),
    )
    parser.add_argument("--verbose", "-v", default=False, action="store_true", help="Verbose output")
    parser.add_argument("--save-dataset", "-s", default=False, action="store_true", help="Save dataset to disk")
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    args = parser.parse_args()

    eval_environment(
        env=args.env,
        env_args=args.env_args,
        env_dir_path="./examples/vf-envs",
        model=args.model,
        api_key_var=args.api_key_var,
        api_base_url=args.api_base_url,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent_requests=args.max_concurrent_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sampling_args=args.sampling_args,
        verbose=args.verbose,
        save_dataset=args.save_dataset,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
    )


if __name__ == "__main__":
    main()

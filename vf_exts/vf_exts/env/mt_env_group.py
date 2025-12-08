"""Reimplementation of EnvGroup which only accepts multi-turn environments."""


import verifiers as vf
from datasets import concatenate_datasets
from openai import AsyncOpenAI


class _MtEnvGroupRubric(vf.Rubric):
    def __init__(self, env_map: dict[str, vf.MultiTurnEnv]):
        super().__init__()
        self.env_map = env_map
        names = set()
        for env in env_map.values():
            names.update(env.rubric.get_reward_func_names())
        self._all_reward_names = sorted(list(names))

    def get_reward_func_names(self) -> list[str]:
        return self._all_reward_names

    async def score_rollouts(
        self,
        prompts: list[vf.Messages],
        completions: list[vf.Messages],
        answers: list[str],
        states: list[vf.State],
        tasks: list[str],
        infos: list[vf.Info],
        **kwargs,
    ) -> vf.RolloutScores:
        env = self.env_map.get(tasks[0])
        if env is None:
            raise ValueError(f"No environment found for task {tasks[0]}.")

        return await env.rubric.score_rollouts(
            prompts, completions, answers, states, tasks, infos, **kwargs
        )

class MultiTurnEnvGroup(vf.MultiTurnEnv):
    """Reimplementation of EnvGroup which only accepts multi-turn environments."""
    def __init__(
        self,
        envs: list[vf.MultiTurnEnv],
        env_names: list[str] | None = None,
        message_type: vf.MessageType | None = None,
        max_turns: int | None = None,
        force_overwrite_task: bool = False,
        **kwargs,
    ):
        """Initialize the multi-turn environment group."""
        if not envs:
            raise ValueError("MultiTurnEnvGroup requires at least one environment")

        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]
        if len(self.env_names) != len(self.envs):
            raise ValueError("Number of env_names must match number of envs")
        self.env_map: dict[str, vf.MultiTurnEnv] = {
            name: env for name, env in zip(self.env_names, self.envs)
        }

        datasets = []
        eval_datasets = []
        for env, name in zip(self.envs, self.env_names):
            def add_task(example):
                example["task"] = name
                return example

            env_dataset = env.get_dataset() if hasattr(env, "get_dataset") else None
            if (env_dataset is not None and "task" not in env_dataset.column_names) or force_overwrite_task:
                env_dataset = env_dataset.map(add_task)
            if env_dataset is not None:
                datasets.append(env_dataset)

            env_eval_dataset = (
                env.get_eval_dataset() if hasattr(env, "get_eval_dataset") else None
            )
            if (
                env_eval_dataset is not None
                and "task" not in env_eval_dataset.column_names
            ):
                env_eval_dataset = env_eval_dataset.map(add_task)
            if env_eval_dataset is not None:
                eval_datasets.append(env_eval_dataset)

        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        rubric = _MtEnvGroupRubric(self.env_map)

        resolved_message_type = (
            message_type if message_type is not None else getattr(envs[0], "message_type", "chat")
        )
        resolved_max_turns = (
            max_turns if max_turns is not None else min(getattr(e, "max_turns", 10) for e in envs)
        )

        super().__init__(
            message_type=resolved_message_type,
            max_turns=resolved_max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            **kwargs,
        )
        self.logger.info(
            f"Initialized MultiTurnEnvGroup with {len(envs)} environments: {self.env_names}"
        )

    def get_env_for_task(self, task: str) -> vf.MultiTurnEnv:
        """Get the environment for a given task."""
        env = self.env_map.get(task, None)
        assert env is not None, f"No environment found for task {task}."
        return env

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Setup the initial state for the selected environment."""
        task = state.get("task", None)
        assert task is not None, "Could not find task in state."
        env = self.get_env_for_task(task)
        state["task"] = task
        return env.setup_state(state, **kwargs)

    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if the environment is completed."""
        task = state.get("task", None)
        assert task is not None, "Could not find task in state."
        env = self.get_env_for_task(task)
        return env.is_completed(messages, state, **kwargs)

    def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> tuple[vf.Messages, vf.State]:
        """Get the environment response for a given messages and state."""
        task = state.get("task", None)
        assert task is not None, "Could not find task in state."
        env = self.get_env_for_task(task)
        return env.env_response(messages, state, **kwargs)

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str | list[vf.ChatMessage],
        answer: str = "",
        task: str = "default",
        info: vf.Info = {},
        sampling_args: vf.SamplingArgs = {},
        **kwargs,
    ) -> tuple[str | list[vf.ChatMessage], vf.State]:
        """Rollout the environment for a given task."""
        env = self.get_env_for_task(task)
        return await env.rollout(client, model, prompt, answer, task, info, sampling_args, **kwargs)

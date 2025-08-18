import verifiers as vf
from typing import Dict, List, Tuple, Union
from datasets import concatenate_datasets
from openai import AsyncOpenAI


class _MtEnvGroupRubric(vf.Rubric):
    def __init__(self, env_map: Dict[str, vf.MultiTurnEnv]):
        super().__init__()
        self.env_map = env_map
        names = set()
        for env in env_map.values():
            names.update(env.rubric.get_reward_func_names())
        self._all_reward_names = sorted(list(names))

    def get_reward_func_names(self) -> List[str]:
        return self._all_reward_names

    async def score_rollout(
        self,
        prompt: Union[str, List[vf.ChatMessage]],
        completion: Union[str, List[vf.ChatMessage]],
        answer: str = "",
        state: vf.State = {},
        task: str = "default",
        info: dict = {},
        **kwargs,
    ) -> vf.RolloutScore:
        metrics = {name: 0.0 for name in self._all_reward_names}
        reward = 0.0
        env = self.env_map.get(task)
        if env is None:
            return vf.RolloutScore(reward=reward, metrics=metrics)
        env_results = await env.rubric.score_rollout(
            prompt, completion, answer, state, task, info, **kwargs
        )
        for reward_name, score in env_results.metrics.items():
            if reward_name in metrics:
                metrics[reward_name] = score
        reward = env_results.reward
        return vf.RolloutScore(reward=reward, metrics=metrics)


class MultiTurnEnvGroup(vf.MultiTurnEnv):
    def __init__(
        self,
        envs: List[vf.MultiTurnEnv],
        env_names: List[str] | None = None,
        message_type: vf.MessageType | None = None,
        max_turns: int | None = None,
        force_overwrite_task: bool = False,
        **kwargs,
    ):
        if not envs:
            raise ValueError("MultiTurnEnvGroup requires at least one environment")

        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]
        if len(self.env_names) != len(self.envs):
            raise ValueError("Number of env_names must match number of envs")
        self.env_map: Dict[str, vf.MultiTurnEnv] = {
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
        return self.env_map.get(task, self.envs[0])

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        task = state.get("task", self.env_names[0])
        env = self.get_env_for_task(task)
        state["task"] = task
        return env.setup_state(state, **kwargs)

    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        task = state.get("task", self.env_names[0])
        env = self.get_env_for_task(task)
        return env.is_completed(messages, state, **kwargs)

    def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> Tuple[vf.Messages, vf.State]:
        task = state.get("task", self.env_names[0])
        env = self.get_env_for_task(task)
        return env.env_response(messages, state, **kwargs)

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Union[str, List[vf.ChatMessage]],
        answer: str = "",
        task: str = "default",
        info: vf.Info = {},
        sampling_args: vf.SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Union[str, List[vf.ChatMessage]], vf.State]:
        env = self.env_map.get(task, self.envs[0])
        return await env.rollout(
            client, model, prompt, answer, task, info, sampling_args, **kwargs
        )
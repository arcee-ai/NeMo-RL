import asyncio
import inspect
import logging
from typing import Callable, Coroutine, List

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RewardFunc, RolloutScores, State

GroupedRewardFunc = Callable[..., list[float]]
class GroupedRubric(Rubric):
    """
    Rubric base class for grouped scoring.

    Reward functions on this rubric operate on a group of rollouts at once.
    Each argument that would normally be a scalar for a single rollout is now
    a list of equal length, and the reward function returns a list of floats
    (one per rollout in the group).
    """

    def __init__(
        self,
        funcs: List[GroupedRewardFunc] = [],
        weights: List[float] = [],
        parser: Parser = Parser(),
        parallelize_scoring: bool = True,
        **kwargs,
    ):
        super().__init__(
            funcs=funcs,
            weights=weights,
            parser=parser,
            parallelize_scoring=parallelize_scoring,
            **kwargs,
        )
        self.logger = logging.getLogger(
            f"verifiers.rubrics.{self.__class__.__name__}"
        )

    async def call_group_reward_func(
        self,
        func: GroupedRewardFunc,
        parser: Parser,
        prompts: List[Messages],
        completions: List[Messages],
        answer: str,
        states: List[State],
        task: str,
        info: Info,
        **kwargs,
    ) -> List[float]:
        """
        Invoke a group-aware reward function with only the required arguments.

        The function is expected to return a list of floats of the same length
        as the provided inputs.
        """
        sig = inspect.signature(func)

        common = dict(
            parser=parser,
            prompts=prompts,
            completions=completions,
            answer=answer,
            state=states,
            task=task,
            info=info,
        )
        merged = {**common, **kwargs}
        try:
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                result = func(**merged)
            else:
                allowed = {k: v for k, v in merged.items() if k in sig.parameters}
                result = func(**allowed)
        except Exception as e:
            self.logger.error(
                f"Error calling grouped reward function {func.__name__}: {e}"
            )
            return [0.0] * len(prompts)

        # Allow reward functions to be async.
        if inspect.isawaitable(result):
            result = await result

        if not isinstance(result, list):
            self.logger.error(
                f"Grouped reward function {func.__name__} did not return a list."
            )
            return [0.0] * len(prompts)

        if len(result) != len(prompts):
            self.logger.error(
                "Length mismatch from grouped reward function %s: expected %d, got %d",
                func.__name__,
                len(prompts),
                len(result),
            )
            return [0.0] * len(prompts)

        try:
            return [float(x) for x in result]
        except Exception:
            self.logger.error(
                f"Grouped reward function {func.__name__} returned non-float values."
            )
            return [0.0] * len(prompts)

    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        max_concurrent: int = -1,
        **kwargs,
    ) -> RolloutScores:
        """
        Compatibility wrapper for old GroupedRubric interface.
        """
        # Sanity check to make sure all of these are from the same prompt.
        first_info = infos[0]
        for info in infos:
            assert info == first_info, "GroupedRubric got rollouts with different infos, implying mixed groups."
        
        first_task = tasks[0]
        for task in tasks:
            assert task == first_task, "GroupedRubric got rollouts with different tasks - note that interleaved grading is not supported."
        
        return await self.score_rollouts_grouped(
            prompts=prompts,
            completions=completions,
            answer=group_answers[0],
            states=states,
            task=tasks[0],
            info=infos[0],
            **kwargs
        )

    def should_group_rollouts(self, task: str) -> bool:
        return True

    async def score_rollouts_grouped(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        answer: str,
        states: List[State],
        task: str,
        info: Info,
        **kwargs,
    ) -> RolloutScores:
        """
        Compute reward scores for a group of rollouts in a single pass per reward
        function. Each reward function receives the full lists and returns a list
        of scores corresponding to each rollout.
        """
        group_size = len(prompts)
        if group_size == 0:
            reward_func_names = self.get_reward_func_names()
            return RolloutScores(
                reward=[],
                metrics={name: [] for name in reward_func_names},
            )

        if self.parallelize_scoring:
            score_tasks = [
                self.call_group_reward_func(
                    func=func,
                    parser=self.parser,
                    prompts=prompts,
                    completions=completions,
                    answer=answer,
                    states=states,
                    task=task,
                    info=info,
                    **kwargs,
                )
                for func in self.get_reward_funcs()
            ]
            reward_scores_by_func = await asyncio.gather(*score_tasks)
        else:
            reward_scores_by_func = []
            for func in self.get_reward_funcs():
                scores = await self.call_group_reward_func(
                    func=func,
                    parser=self.parser,
                    prompts=prompts,
                    completions=completions,
                    answer=answer,
                    states=states,
                    task=task,
                    info=info,
                    **kwargs,
                )
                reward_scores_by_func.append(scores)

        metrics = {
            func.__name__: scores
            for func, scores in zip(self.get_reward_funcs(), reward_scores_by_func)
        }

        weights = self.get_reward_weights()
        combined_reward = [
            sum(score * weight for score, weight in zip(scores_at_i, weights))
            for scores_at_i in zip(*reward_scores_by_func)
        ]

        return RolloutScores(reward=combined_reward, metrics=metrics)



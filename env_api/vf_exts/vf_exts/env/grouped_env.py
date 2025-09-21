from copy import deepcopy
import verifiers as vf

def split_rollouts_by_group(
    prompts: list[Messages],
    results: list[tuple[Messages, State]],
    grpo_gids: list[int],
) -> dict[int, list[tuple[Messages, Messages, State]]]:
    """
    Split rollouts by GRPO group.

    Args:
        prompts: List of prompts.
        results: List of tuples of (completion, state).
        grpo_gids: List of GRPO group IDs.

    Returns:
        Dictionary of GRPO group IDs to lists of tuples of (prompt, completion, state).
    """
    results_by_group = {}
    for prompt, result, grpo_gid in zip(prompts, results, grpo_gids):
        if grpo_gid not in results_by_group:
            results_by_group[grpo_gid] = []
        results_by_group[grpo_gid].append((prompt, result[0], result[1]))
    return results_by_group

class GroupedEnv(vf.MultiTurnEnv):
    """
    A MultiTurnEnv that provides GRPO groups for group-relative grading.
    """
    async def run_rollouts(
        self,
        client: AsyncOpenAI,
        model: str,
        prompts: list[Messages],
        answers: list[str],
        tasks: list[str],
        infos: list[Info],
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        # Custom kwargs
        grpo_gids: list[int] | None = None, # List the same size 
        **kwargs,
    ) -> list[tuple[Messages, State]]:
        assert grpo_gids is not None, "GRPO GIDs must be provided for GroupedEnv."
        assert len(grpo_gids) == len(prompts), "GRPO GIDs must be the same size as the number of prompts."
        results = await super().run_rollouts(
            client, model, prompts, answers, tasks, infos, sampling_args, max_concurrent, **kwargs
        )

        # Split up results by GRPO group.
        results_by_group = split_rollouts_by_group(prompts, results, grpo_gids)

        # Add group index and group to state.
        for grpo_gid, group in results_by_group.items():
            for i, (prompt, completion, state) in enumerate(group):
                state["grpo_group_index"] = i
                state["grpo_group"] = group
        
        return results
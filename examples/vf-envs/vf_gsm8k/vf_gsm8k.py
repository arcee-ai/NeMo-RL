"""GSM8K single-turn math environment using verifiers (grouped scoring).

Public objects:
- load_environment

External dependencies:
- verifiers

Usage example:
    import verifiers as vf
    # Either name works depending on loader resolution
    env = vf.load_environment("vf_gsm8k")  # or "vf-gsm8k"
    _ = env.reset()
"""

from __future__ import annotations

from pocketReward import getReward

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)
import vf_exts as vfe


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
) -> vf.Environment:
    """Construct a GSM8K SingleTurnEnv with grouped rubric (batched scoring).

    Args:
        use_think (bool): If True, use ThinkParser; otherwise use Parser.
        system_prompt (str | None): If None, defaults to THINK_BOXED_SYSTEM_PROMPT when
            use_think is True, else BOXED_SYSTEM_PROMPT.
        num_train_examples (int): Optional cap on training examples; -1 uses all.
        num_eval_examples (int): Optional cap on eval examples; -1 uses all.

    Returns:
        vf.Environment: A configured SingleTurnEnv for GSM8K.
    """
    # Choose a system prompt consistent with the parser behavior when not provided.
    if system_prompt is None:
        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))

    parser: vf.Parser
    # NOTE: Trim whitespace around extracted boxed content to avoid false negatives
    # when comparing against GSM8K ground-truth answers (which are stripped).
    if use_think:
        parser = vf.ThinkParser(extract_fn=lambda s: extract_boxed_answer(s).strip())
    else:
        parser = vf.Parser(extract_fn=lambda s: extract_boxed_answer(s).strip())

    def correctAnswerRewardFunc(
        prompts: list[vf.Messages],
        completions: list[vf.Messages],
        answer: str,
        parser: vf.Parser,
        **kwargs,
    ) -> list[float]:
        """Return 1.0 if parsed completion matches ground-truth; else 0.0, per item."""
        rewards: list[float] = []
        normalized_answer = (answer or "").strip()
        for comp in completions:
            # Extract final assistant message content if available.
            content = ""
            if isinstance(comp, list) and len(comp) > 0:
                # Find last assistant message; default to last element.
                for m in reversed(comp):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        content = m.get("content", "")
                        break
                else:
                    content = comp[-1].get("content", "") if isinstance(comp[-1], dict) else str(comp[-1])
            elif isinstance(comp, str):
                content = comp
            parsed = parser.parse_answer(content) or ""
            rewards.append(1.0 if parsed == normalized_answer else 0.0)
        return rewards

    async def externalRewardFunc(
        prompts: list[vf.Messages],
        completions: list[vf.Messages],
        **kwargs,
    ) -> list[float]:
        """Call pocketReward sequentially over the group and return per-item scores."""
        async def score_one(prompt: vf.Messages, completion: vf.Messages) -> float:
            messages: vf.Messages = []
            if isinstance(prompt, list):
                messages += list(prompt)
            if isinstance(completion, list):
                messages += list(completion)
            elif isinstance(completion, str) and completion:
                messages.append({"role": "assistant", "content": completion})
            # Determine whether stripping the completion would change its content.
            # NOTE: We only penalize the score; we do not modify the messages sent to the scorer.
            strip_changed: bool = False
            if isinstance(completion, str):
                strip_changed = completion.strip() != completion
            elif isinstance(completion, list):
                for m in completion:
                    content = m.get("content") if isinstance(m, dict) else None
                    if isinstance(content, str) and content.strip() != content:
                        strip_changed = True
                        break

            result = await getReward(messages=messages)
            try:
                score = float(result)
            except Exception:
                return 0.0

            if strip_changed:
                # NOTE: Apply a 10% penalty when extraneous surrounding whitespace is present.
                score *= 0.9

            return score
        
        results: list[float] = []
        for p, c in zip(prompts, completions):
            results.append(await score_one(p, c))
        return results

    def formatRewardFunc(
        completions: list[vf.Messages],
        parser: vf.Parser,
        **kwargs,
    ) -> list[float]:
        """Grouped wrapper around parser-provided format reward function."""
        base = parser.get_format_reward_func()
        scores: list[float] = []
        for comp in completions:
            try:
                scores.append(float(base(comp)))
            except Exception:
                scores.append(0.0)
        return scores

    rubric = vfe.GroupedRubric(
        funcs=[correctAnswerRewardFunc, externalRewardFunc, formatRewardFunc],
        weights=[0.7, 0.1, 0.2],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return env

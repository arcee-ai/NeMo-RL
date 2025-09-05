"""BasicQA single-turn environment using verifiers (grouped scoring).

Public objects:
- load_environment

External dependencies:
- verifiers

Usage example:
    import verifiers as vf
    # Either name works depending on loader resolution
    env = vf.load_environment("vf_basicqa", system_prompt="You are an instance of AFM v1.1, aka the Arcee Foundation Model. You were trained by Arcee AI.")  # or "vf-basicqa"
    _ = env.reset()
"""

from __future__ import annotations

from pocketReward import getReward
import datasets

import verifiers as vf
import vf_exts as vfe

def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = "The assistant is AFM-4.5B, trained by Arcee AI, with 4.5 billion parameters. AFM is a deeply thoughtful, helpful assistant. The assistant is having a conversation with the user. The assistant's responses are calm, intelligent, and personable, always aiming to truly understand the user's intent. AFM thinks aloud, step by step, when solving problems or forming explanations, much like a careful, reflective thinker would. The assistant helps with sincerity and depth. If a topic invites introspection, curiosity, or broader insight, the assistant allows space for reflection — be open to nuance and complexity. The assistant is not robotic or overly formal; it speaks like a wise, thoughtful companion who cares about clarity and the human experience. If a topic is uncertain or depends on subjective interpretation, AFM explains the possibilities thoughtfully.",
) -> vf.Environment:
    """Construct a BasicQA SingleTurnEnv with grouped rubric (batched scoring).

    Args:
        num_train_examples (int): Optional cap on training examples; -1 uses all.
        num_eval_examples (int): Optional cap on eval examples; -1 uses all.
        system_prompt (str): System prompt to prepend to the conversation.

    Returns:
        vf.Environment: A configured SingleTurnEnv for GSM8K.
    """
    dataset = datasets.load_dataset("arcee-train/pocket-lmsysPrompts", split="train")

    # Drop any samples with a prompt longer than 4000 characters
    dataset = dataset.filter(lambda x: len(x["prompt"]) < 4000)

    # Shuffle with seed 42
    dataset = dataset.shuffle(seed=42)
    
    # The prompts are all in the key "prompt"
    dataset = dataset.map(
        lambda x, i: {
            "prompt": (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["prompt"]},
                ]
                if (i % 2 == 0)
                else [
                    {"role": "user", "content": x["prompt"]},
                ]
            ),
            "answer": "N/A",
        },
        with_indices=True,
    )

    # Create an eval holdout if requested; otherwise eval over full dataset
    if num_eval_examples != -1:
        take = min(num_eval_examples, len(dataset))
        eval_dataset = dataset.select(range(take))
        dataset = dataset.select(range(take, len(dataset)))
    else:
        eval_dataset = dataset.select(range(len(dataset)))

    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    async def externalRewardFunc(
        prompts: list[vf.Messages],
        completions: list[vf.Messages],
        **kwargs,
    ) -> list[float]:
        """Call pocketReward sequentially over the group and return per-item scores."""
        async def score_one(prompt: vf.Messages, completion: vf.Messages) -> float:
            messages: vf.Messages = []
            # Add system prompt
            if isinstance(prompt, list):
                messages += list(prompt)
            if isinstance(completion, list):
                messages += list(completion)
            elif isinstance(completion, str) and completion:
                messages.append({"role": "assistant", "content": completion})

            result = await getReward(messages=messages)
            try:
                score = float(result)
            except Exception:
                return 0.0

            return score
        
        results: list[float] = []
        for p, c in zip(prompts, completions):
            results.append(await score_one(p, c))
        return results
    

    rubric = vfe.GroupedRubric(
        funcs=[externalRewardFunc],
        weights=[1.0],
    )

    env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )
    return env

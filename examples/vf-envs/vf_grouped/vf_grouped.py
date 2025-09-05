import verifiers as vf
import vf_exts as vfe

import datasets
from datasets import concatenate_datasets

import asyncio
import os
import re
import logging
import html

import openai

from pocketReward import getReward
from collections import Counter

JUDGE_PROMPT = """
You are a strict verifier. Compare the candidate answer with the ground truth.

Return only the XML block exactly in the prescribed format.

Set <score> to an integer 0-5.

Use this rubric:
5: Perfect semantic match; precise, complete, no contradictions.
4: Strong alignment with minor omissions or wording differences.
3: Partially correct; captures key elements but misses others.
2: Weak alignment; vague or significant errors vs. ground truth.
1: Barely related; mostly incorrect with minor overlap.
0: Unrelated or contradicts the ground truth.
Do not include any text outside the block.

Output format:

<output>

<reasoning>
...
</reasoning>

<score>
...
</score>

</output>
"""

async def get_opinion(client: openai.AsyncOpenAI, prompt: vf.Messages, completion: vf.Messages, answer: str) -> int:
    response = await client.chat.completions.create(
        model="moonshotai/kimi-k2-0905:nitro",
        messages=[
            {"role": "user", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"<prompt>\n{prompt}\n</prompt>\n\n"
                f"<candidateAnswer>\n{completion}\n</candidateAnswer>\n\n"
                f"<groundTruth>\n{answer}\n</groundTruth>\n\n"}
        ]
    )
        
    content = response.choices[0].message.content or ""
    # Unescape in case the model returns HTML-escaped tags like &lt;score&gt;0&lt;/score&gt;.
    content_unescaped = html.unescape(content)
    last_match = None
    for m in re.finditer(r"<\s*score\s*>\s*([0-5])\s*<\s*/\s*score\s*>", content_unescaped, re.IGNORECASE):
        last_match = m
    if last_match is None:
        logging.error(f"No answer found in response: {content!r}")
        return 3
    return int(last_match.group(1))

def load_environment(num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 42,
    system_prompt: str = "Thoroughly think about and check your answer before calling it final.",
) -> vf.Environment:
    dataset = datasets.load_dataset("arcee-train/pocket-verifiableQaPrepped", split="train")

    # Drop any samples with a prompt longer than 4000 characters
    dataset = dataset.filter(lambda x: len(x["prompt"]) < 4000)

    # Shuffle with seed 42
    dataset = dataset.shuffle(seed=seed)

    # The dataset has a key "domain". Compute the average count per domain and
    # select up to that many examples for each domain without dropping other fields.
    domain_counts = Counter(dataset["domain"])
    avg_count = int(sum(domain_counts.values()) / len(domain_counts))
    kept_per_domain: dict[str, int] = {}
    keep_indices: list[int] = []
    for idx, d in enumerate(dataset["domain"]):
        c = kept_per_domain.get(d, 0)
        if c < avg_count:
            keep_indices.append(idx)
            kept_per_domain[d] = c + 1
    dataset = dataset.select(keep_indices)

    # Add arcee-train/pocket-practicePairsJoined
    practice_dataset = datasets.load_dataset("arcee-train/pocket-practicePairsJoined", split="train")

    # Drop any rows with prompts or answers that contain "fig" case-insensitively
    practice_dataset = practice_dataset.filter(lambda x: "fig" not in x["prompt"].lower() and "fig" not in x["groundTruth"].lower())
    
    # Drop any samples with a prompt longer than 4000 characters
    practice_dataset = practice_dataset.filter(lambda x: len(x["prompt"]) < 4000)

    dataset = concatenate_datasets([dataset, practice_dataset])

    # Shuffle with seed 42
    dataset = dataset.shuffle(seed=seed)

    # The prompts are all in the key "prompt"
    dataset = dataset.map(
        lambda x, i: {
            "prompt": [
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["groundTruth"],
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

    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    client = openai.AsyncOpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
    
    async def verifiableQaFunc(prompts: list[vf.Messages], completions: list[vf.Messages], answer: str, **kwargs) -> list[float]:
        rewards = []

        tasks = [get_opinion(client, a, b, answer) for a, b in zip(prompts, completions)]
        scores = await asyncio.gather(*tasks)

        for score in scores:
            rewards.append(score / 5)

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
    
    rubric = vfe.GroupedRubric(
        funcs=[verifiableQaFunc, externalRewardFunc],
        weights=[0.7, 0.3],
    )
    
    env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )
    return env
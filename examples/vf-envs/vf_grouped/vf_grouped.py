import verifiers as vf
import vf_exts as vfe

from datasets import Dataset

import asyncio
import random
import os
import math
import re
import logging

import openai

various_nouns = [
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "rabbit",
    "snake",
    "tiger",
    "chair",
    "table",
    "book",
    "pen",
    "pencil",
    "eraser",
    "notebook",
    "computer",
    "program",
    "ai",
    "robot"
]

JUDGE_PROMPT = """
You are a poet. You are given two poems. Please judge which poem is better.

Respond with an integer between 1 and 7 (inclusive) indicating your preference, where 1 is "Poem A is much better than Poem B", 4 is "Poem A and Poem B are equally good", and 7 is "Poem B is much better than Poem A".

Put your judgement in an <answer> tag, like this:

<answer>5</answer>

Do not include any other text in your response.

Poem A:
{}

Poem B:
{}
"""

async def get_opinion(client: openai.AsyncOpenAI, poem_a: str, poem_b: str) -> int:
    response = await client.chat.completions.create(
        model="moonshotai/kimi-k2",
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(poem_a, poem_b)}
        ]
    )
    
    match = re.search(r"<answer>(.*?)</answer>", response.choices[0].message.content)
    if match is None:
        logging.error(f"No answer found in response: {response.choices[0].message.content}")
        return 4
    return int(match.group(1))

def load_environment(num_examples=100,seed=42) -> vf.MultiTurnEnv:
    random.seed(seed)
    dataset = Dataset.from_dict({
        "question": [f"Please write a poem about the following word: {random.choice(various_nouns)}" for i in range(num_examples)],
        "answer": ["doesn't matter"] * num_examples,
    })
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    client = openai.AsyncOpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
    
    async def poem_reward_func(completions: list[vf.Messages], answer: list[str], **kwargs) -> list[float]:
        rewards = []
        for i in range(0, len(completions), 2):
            a = completions[i][-1]["content"]
            b = completions[i+1][-1]["content"]
            opinion = await get_opinion(client, a, b)
            
            b_reward = opinion / 7
            a_reward = 1 - b_reward
            rewards.append(a_reward)
            rewards.append(b_reward)
        
        return rewards
    
    rubric = vfe.GroupedRubric([poem_reward_func])
    
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env
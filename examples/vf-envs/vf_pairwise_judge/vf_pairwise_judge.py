import verifiers as vf
import vf_exts as vfe

from datasets import Dataset

import random
import os

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

def load_environment(num_examples=100,seed=42) -> vf.MultiTurnEnv:
    random.seed(seed)
    dataset = Dataset.from_dict({
        "question": [f"Please write a poem about the following word: {random.choice(various_nouns)}" for i in range(num_examples)],
        "answer": ["doesn't matter"] * num_examples,
    })
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    client = openai.OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
    
    rubric = vfe.PairwiseJudgeRubric(judge_client=client, judge_model="google/gemini-2.5-flash-lite")
    
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env
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

def load_environment(num_examples=1000, seed=42) -> vf.MultiTurnEnv:
    random.seed(seed)
    dataset = Dataset.from_dict({
        "question": [f"Please write a poem about the following word: {random.choice(various_nouns)}" for i in range(num_examples)],
        "answer": [""] * num_examples,
    })
    
    client = openai.OpenAI(api_key="n/a", base_url="http://127.0.0.1:8000/v1")
    
    rubric = vfe.PairwiseJudgeRubric(judge_client=client, judge_model="policy", rubric_prompt="The poem should include as many markdown formatting elements as humanly possible.")
    
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env
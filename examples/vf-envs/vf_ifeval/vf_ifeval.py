import vf_exts as vfe

from datasets import load_dataset
from ifeval_utils.ifeval_singleturn import IFEvalSingleTurnEnv
from ifeval_utils.ifeval_rubric import IF_FUNCTIONS_MAP
import json
from pocketReward import getReward
from verifiers.types import Messages

def load_environment(num_train_examples: int = -1, num_eval_examples: int = -1):
    """IFEval environment using custom single-turn chat env and rubric.

    - Loads allenai/RLVR-IFeval
    - Maps to chat-style prompt (messages list) and JSON string answer expected by IFEvalRubric
    - Uses IFEvalSingleTurnEnv and IFEvalRubric
    """
    ds = load_dataset("allenai/RLVR-IFeval")

    # Drop any samples with a prompt longer than 4000 characters
    ds = ds.filter(lambda x: len(x["messages"][0]["content"]) < 4000)

    # Shared system prompt for this environment
    system_prompt = "Answer the following question."

    async def ifevalExternalRewardGrouped(
        prompts: list[Messages], completions: list[Messages], **kwargs
    ) -> list[float]:
        """Sequentially call pocketReward for each (prompt, completion) pair."""
        results: list[float] = []
        for prompt, completion in zip(prompts, completions):
            messages = list(prompt) if isinstance(prompt, list) else []
            if isinstance(completion, list):
                messages += completion
            elif isinstance(completion, str) and completion:
                messages.append({"role": "assistant", "content": completion})
            try:
                score = await getReward(messages=messages)
                results.append(float(score))
            except Exception:
                results.append(0.0)
        return results

    def ifevalRewardGrouped(
        completions: list[Messages], answer: str | list[str], **kwargs
    ) -> list[float]:
        """Compute IFEval rule scores per completion using IF_FUNCTIONS_MAP.

        Accepts 'answer' as either a JSON string or a list of JSON strings.
        """
        def extract_response(comp) -> str:
            if isinstance(comp, list):
                # Try to get the last assistant message content; fallback to string.
                for m in reversed(comp):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        return str(m.get("content", "")).strip()
                return str(comp[-1]).strip() if comp else ""
            if isinstance(comp, str):
                return comp.strip()
            return str(comp).strip()

        def get_answer_for_index(ans, idx: int) -> str:
            if isinstance(ans, list):
                if len(ans) == 0:
                    return "{}"
                return ans[idx] if idx < len(ans) else ans[-1]
            return ans

        scores: list[float] = []
        for i, comp in enumerate(completions):
            response = extract_response(comp)
            ans_i = get_answer_for_index(answer, i)
            try:
                constraint_dict = json.loads(ans_i)
            except Exception:
                scores.append(0.0)
                continue

            func_name = constraint_dict.pop("func_name", None)
            if not func_name:
                scores.append(0.0)
                continue
            func = IF_FUNCTIONS_MAP.get(func_name)
            if func is None:
                scores.append(0.0)
                continue
            non_none_args = {k: v for k, v in constraint_dict.items() if v is not None}
            try:
                val = func(response, **non_none_args) if non_none_args else func(response)
                scores.append(float(val))
            except Exception:
                scores.append(0.0)
        return scores

        
    def _map_example(ex):
        # Question comes from first user message in messages
        question = ex.get("messages", [{}])[0].get("content", "")
        # Ground truth is a JSON string with constraint and args
        answer = ex.get("ground_truth", "{}")
        messages = [
            {"role": "user", "content": question},
        ]
        ex["prompt"] = messages
        ex["answer"] = answer
        return ex

    train_split = ds.get("train") or ds[list(ds.keys())[0]]
    # Determine eval split; if missing, create a holdout from train
    if "validation" in ds:
        eval_split = ds["validation"]
    elif "test" in ds:
        eval_split = ds["test"]
    else:
        # Create an eval holdout from train. Use requested num_eval_examples when provided.
        if num_eval_examples != -1:
            eval_size = max(
                1,
                min(
                    num_eval_examples,
                    len(train_split) - 1 if len(train_split) > 1 else 1,
                ),
            )
        else:
            # Default to min(200, 10% of train)
            eval_size = max(1, min(200, max(1, len(train_split) // 10)))
        split = train_split.train_test_split(test_size=eval_size, seed=42, shuffle=True)
        train_split = split["train"]
        eval_split = split["test"]

    # Map to prompt/answer expected by the rubric/env
    train_split = train_split.map(_map_example)
    eval_split = eval_split.map(_map_example)

    # Optional subsampling after mapping
    if num_train_examples != -1:
        train_split = train_split.select(
            range(min(num_train_examples, len(train_split)))
        )
    if num_eval_examples != -1:
        eval_split = eval_split.select(range(min(num_eval_examples, len(eval_split))))

    rubric = vfe.GroupedRubric(
        funcs=[ifevalRewardGrouped, ifevalExternalRewardGrouped],
        weights=[0.9, 0.1],
    )

    env = IFEvalSingleTurnEnv(
        dataset=train_split,
        eval_dataset=eval_split,
        parser=None,
        system_prompt=system_prompt,
        rubric=rubric,
        message_type="chat",
    )
    return env

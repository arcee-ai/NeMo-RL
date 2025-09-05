"""Sandboxed code verification environment.

Public objects:
- load_environment
- verifyCode
- runPythonInSandbox

External dependencies:
- datasets, verifiers, vf_exts, docker

Usage example:
    env = load_environment(num_train_examples=10, num_eval_examples=10)
"""
import asyncio
import re
import math
import random
import base64
import threading
import hashlib
from typing import Dict, List
import json
import io
import tarfile

from pocketReward import getReward

import datasets
import verifiers as vf

import vf_exts as vfe

try:
    import docker
except Exception:  # noqa: BLE001
    docker = None

_docker_client = None
_container_pool: list[tuple[object, threading.Lock]] = []
_pool_index = 0
_pool_index_lock = threading.Lock()
POOL_SIZE = 20  # Default parallelism for execs


def callPythonScriptDocker(script: str, stdinData: str, timeoutSecs: int = 5) -> str:
    """Execute Python code in a sandboxed Docker container.

    Args:
        script (str): Python source code.
        stdinData (str): Content to pipe to stdin.
        timeoutSecs (int): Wall-clock execution limit in seconds.

    Returns:
        str: Combined stdout/stderr text.

    Raises:
        RuntimeError: If docker SDK is unavailable.
    """
    global _docker_client, _container_pool, _pool_index
    if docker is None:
        raise RuntimeError("docker Python SDK is not installed.")

    # Lazy-init docker client and ensure image exists.
    if _docker_client is None:
        _docker_client = docker.from_env()
        try:
            _docker_client.images.get("python:3.11")
        except Exception:  # noqa: BLE001
            _docker_client.images.pull("python:3.11")

    # Ensure a pool of reusable containers exists.
    if not _container_pool:
        for _ in range(POOL_SIZE):
            c = _docker_client.containers.create(
                "python:3.11",
                command="sleep infinity",
                detach=True,
                network_disabled=True,
                mem_limit="128m",
                nano_cpus=1_000_000_000,
            )
            c.start()
            _container_pool.append((c, threading.Lock()))

    # Round-robin select a container and lock to serialize within that container.
    with _pool_index_lock:
        idx = _pool_index % len(_container_pool)
        _pool_index += 1
    container, cx_lock = _container_pool[idx]

    # If container died, recreate in-place.
    try:
        container.reload()
        if container.status != "running":
            try:
                container.remove(force=True)
            except Exception:
                pass
            new_c = _docker_client.containers.create(
                "python:3.11",
                command="sleep infinity",
                detach=True,
                network_disabled=True,
                mem_limit="128m",
                nano_cpus=1_000_000_000,
            )
            new_c.start()
            _container_pool[idx] = (new_c, cx_lock)
            container = new_c
    except Exception:
        try:
            container.remove(force=True)
        except Exception:
            pass
        new_c = _docker_client.containers.create(
            "python:3.11",
            command="sleep infinity",
            detach=True,
            network_disabled=True,
            mem_limit="128m",
            nano_cpus=1_000_000_000,
        )
        new_c.start()
        _container_pool[idx] = (new_c, cx_lock)
        container = new_c

    # Prepare files using base64 heredocs to avoid tar/copy overhead.
    # We rely on external `timeout` to enforce the runtime budget wall-clock.
    wrapped_script = script
    script_b64 = base64.b64encode(wrapped_script.encode("utf-8")).decode("ascii")
    stdin_b64 = base64.b64encode(stdinData.encode("utf-8")).decode("ascii")

    sh = (
        "sh -lc "
        + repr(
            "set -e\n"
            "cat <<'__SCRIPT__' | base64 -d > /tmp/script.py\n"
            f"{script_b64}\n"
            "__SCRIPT__\n"
            "cat <<'__STDIN__' | base64 -d > /tmp/stdin.txt\n"
            f"{stdin_b64}\n"
            "__STDIN__\n"
            f"timeout -s KILL {timeoutSecs}s python /tmp/script.py < /tmp/stdin.txt\n"
        )
    )

    with cx_lock:
        exec_result = container.exec_run(sh, stdout=True, stderr=True)
    output = exec_result.output.decode("utf-8", errors="replace")
    return output

def runPythonInSandbox(script: str, stdinData: str, timeoutSecs: int = 5) -> str:
    """Execute Python code in a Docker sandbox.

    Args:
        script (str): Python source code.
        stdinData (str): Content to pipe to stdin.
        timeoutSecs (int): Runtime budget in seconds.

    Returns:
        str: Program output (stdout preferred).
    """
    return callPythonScriptDocker(script, stdinData, timeoutSecs)


def runPythonBatchInSandbox(script: str, inputs: List[str], timeoutSecs: int = 5) -> List[str]:
    """Execute the same Python script multiple times (one per input) in a single exec.

    Runs a small harness inside the container that launches the user script as a
    subprocess per test case with an individual timeout. This avoids creating a
    new docker exec for each test.

    Args:
        script (str): User Python source code.
        inputs (list[str]): Stdin payloads for each test case.
        timeoutSecs (int): Per-test wall-clock limit.

    Returns:
        list[str]: Combined stdout/stderr for each test case, same length as inputs.
    """
    global _docker_client, _container_pool, _pool_index
    if docker is None:
        raise RuntimeError("docker Python SDK is not installed.")

    # Ensure docker client and container pool exist.
    if _docker_client is None:
        _docker_client = docker.from_env()
        try:
            _docker_client.images.get("python:3.11")
        except Exception:  # noqa: BLE001
            _docker_client.images.pull("python:3.11")
    if not _container_pool:
        for _ in range(POOL_SIZE):
            c = _docker_client.containers.create(
                "python:3.11",
                command="sleep infinity",
                detach=True,
                network_disabled=True,
                mem_limit="128m",
                nano_cpus=1_000_000_000,
            )
            c.start()
            _container_pool.append((c, threading.Lock()))

    # Round-robin select a container and lock.
    with _pool_index_lock:
        idx = _pool_index % len(_container_pool)
        _pool_index += 1
    container, cx_lock = _container_pool[idx]

    # Ensure selected container is running.
    try:
        container.reload()
        if container.status != "running":
            try:
                container.remove(force=True)
            except Exception:
                pass
            new_c = _docker_client.containers.create(
                "python:3.11",
                command="sleep infinity",
                detach=True,
                network_disabled=True,
                mem_limit="128m",
                nano_cpus=1_000_000_000,
            )
            new_c.start()
            _container_pool[idx] = (new_c, cx_lock)
            container = new_c
    except Exception:
        try:
            container.remove(force=True)
        except Exception:
            pass
        new_c = _docker_client.containers.create(
            "python:3.11",
            command="sleep infinity",
            detach=True,
            network_disabled=True,
            mem_limit="128m",
            nano_cpus=1_000_000_000,
        )
        new_c.start()
        _container_pool[idx] = (new_c, cx_lock)
        container = new_c

    # Prepare files (script.py, inputs.json, harness.py) via tar put_archive for reliability.
    inputs_json = json.dumps([str(s) for s in inputs], ensure_ascii=False)
    harness_code = (
        "import json, subprocess, sys\n"
        "with open('/tmp/inputs.json','r', encoding='utf-8') as f:\n"
        "    inputs = json.load(f)\n"
        "outs=[]\n"
        f"for s in inputs:\n"
        f"    try:\n"
        f"        p = subprocess.run([sys.executable,'/tmp/script.py'], input=s, capture_output=True, text=True, timeout={timeoutSecs})\n"
        f"        out = (p.stdout or '') + (p.stderr or '')\n"
        f"    except Exception as e:\n"
        f"        out = ''\n"
        f"    outs.append(out)\n"
        "print(json.dumps(outs), end='')\n"
    )

    tarbuf = io.BytesIO()
    with tarfile.open(fileobj=tarbuf, mode="w") as tar:
        # script.py
        script_bytes = script.encode("utf-8")
        ti = tarfile.TarInfo(name="script.py")
        ti.size = len(script_bytes)
        tar.addfile(ti, io.BytesIO(script_bytes))
        # inputs.json
        inputs_bytes = inputs_json.encode("utf-8")
        ti = tarfile.TarInfo(name="inputs.json")
        ti.size = len(inputs_bytes)
        tar.addfile(ti, io.BytesIO(inputs_bytes))
        # harness.py
        harness_bytes = harness_code.encode("utf-8")
        ti = tarfile.TarInfo(name="harness.py")
        ti.size = len(harness_bytes)
        tar.addfile(ti, io.BytesIO(harness_bytes))
    tarbuf.seek(0)

    with cx_lock:
        # Copy tar archive into /tmp inside the container.
        container.put_archive(path="/tmp", data=tarbuf.read())
        # Execute the harness.
        exec_result = container.exec_run("python -S -B /tmp/harness.py", stdout=True, stderr=True)

    out = exec_result.output.decode("utf-8", errors="replace")
    try:
        return json.loads(out)
    except Exception:
        return [""] * len(inputs)


def extractLastCodeBlock(text: str) -> str:
    """Extract the last Python fenced code block; fallback to raw text.

    Args:
        text (str): Text which may contain one or more fenced code blocks.

    Returns:
        str: The extracted Python code, or the original text if none found.
    """
    pattern = r"```(?:python|py)?\s*(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


async def verifyCode(
    completion: vf.Messages,
    testCases: List[Dict[str, str]],
    timeoutSecs: int = 5,
) -> float:
    """Run completion code against stdin/stdout test cases and return pass count.

    Args:
        prompt (vf.Messages): The original prompt messages (unused).
        completion (vf.Messages): Assistant messages containing code.
        testCases (list[dict[str, str]]): Each case has 'input' and 'output'.
        timeoutSecs (int): Execution timeout per test in seconds.

    Returns:
        float: Number of passed test cases.
    """
    # Get most recent assistant content.
    if isinstance(completion, list) and completion:
        last = completion[-1]
        code_text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        code_text = str(completion)

    script = extractLastCodeBlock(code_text)
    # Run all cases for this completion in one container exec for speed.
    inputs = [str(c.get("input", "")) for c in testCases]
    observed_list = await asyncio.to_thread(
        runPythonBatchInSandbox, script, inputs, timeoutSecs
    )
    passes = 0.0
    for case, observed in zip(testCases, observed_list):
        expected = str(case.get("output", ""))
        if str(observed).strip() == expected.strip():
            passes += 1.0
    return passes



def load_environment(num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    dataset = datasets.load_dataset("arcee-train/pocket-taco", split="train")

    # Drop anything that has less than 10 test cases
    dataset = dataset.filter(lambda x: len(x["testCases"]) >= 10)

    # Take up to 3 random examples per row (combined input+output <= 500 chars) and
    # append them to the prompt as plain text examples.
    def _appendExamples(row: Dict[str, object]) -> Dict[str, object]:
        maxChars = 500
        cases = row["testCases"]
        filtered: List[Dict[str, str]] = []
        for case in cases:
            inp = str(case.get("input", ""))
            out = str(case.get("output", ""))
            # Filter out any example exceeding the character budget.
            if len(inp) + len(out) <= maxChars:
                filtered.append({"input": inp, "output": out})

        k = min(3, len(filtered))
        selected = random.sample(filtered, k=k) if k > 0 else []

        if selected:
            exampleLines = [f"Input: {c['input']}\nOutput: {c['output']}" for c in selected]
            examplesText = "\n".join(exampleLines)
            promptWithExamples = f"{row['prompt']}\n\nExamples:\n{examplesText}"
        else:
            promptWithExamples = row["prompt"]

        return {"prompt": promptWithExamples, "answer": json.dumps(row['testCases'])}

    dataset = dataset.map(_appendExamples)

    # Drop any samples that have a prompt longer than 4000 characters
    dataset = dataset.filter(lambda x: len(x["prompt"]) <= 4000)

    # The prompts are all in the key "prompt"; sample a deterministic random 20 answers
    def _wrapAndSample(row: Dict[str, object]) -> Dict[str, object]:
        raw = row.get("answer", "[]")
        try:
            answers = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            answers = []
        if isinstance(answers, list) and len(answers) > 20:
            seed_material = f"{seed}|{row.get('prompt', '')}"
            seed_int = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed_int)
            selected = rng.sample(answers, 20)
        else:
            selected = answers
        return {"prompt": [
            {"role": "system", "content": "Solve the following problem or task using python. Place your final submission in the final code block. Only the standard python packages are available. The script must interact using stdin and stdout."},
            {"role": "user", "content": row["prompt"]}
        ], "answer": json.dumps(selected)}

    dataset = dataset.map(_wrapAndSample)

    dataset = dataset.shuffle(seed=seed)



    # Create an eval holdout if requested; otherwise eval over full dataset
    if num_eval_examples != -1:
        take = min(num_eval_examples, len(dataset))
        eval_dataset = dataset.select(range(take))
        dataset = dataset.select(range(take, len(dataset)))
    else:
        eval_dataset = dataset.select(range(len(dataset)))

    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    
    async def codeVerificationFunc(prompts: list[vf.Messages], completions: list[vf.Messages], answer: str, **kwargs) -> list[float]:
        rewards = []

        # NOTE: verifyCode signature is (completion, testCases, timeoutSecs=5).
        # 'answer' is a JSON string of test cases; parse it and pass as testCases.
        tasks = [verifyCode(completion, json.loads(answer)) for completion in completions]
        
        scores = await asyncio.gather(*tasks)

        eps = 1e-6
        total_cases = len(json.loads(answer)) if answer else 0
        for score in scores:
            # Guard against zero test cases.
            p = (score / total_cases) if total_cases > 0 else 0.0
            failRate = max(eps, 1.0 - p)
            reward = -math.log(failRate) / -math.log(eps)
            rewards.append(reward)

        return rewards

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
        funcs=[codeVerificationFunc, externalRewardFunc],
        weights=[0.9, 0.1],
    )
    
    env = vf.SingleTurnEnv(dataset=dataset, eval_dataset=eval_dataset, rubric=rubric)
    return env
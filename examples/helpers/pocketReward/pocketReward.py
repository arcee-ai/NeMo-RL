"""
Minimal client to score conversations via a remote /classify endpoint.

Public Objects:
    getReward (function)
    getRewards (function)
    DEFAULT_REMOTE_URL (constant)
    DEFAULT_REQUEST_TIMEOUT (constant)
    DEFAULT_CLASSIFY_TIMEOUT (constant)

External Dependencies:
    aiohttp, asyncio, transformers (AutoTokenizer)

Usage example:
    from pocketReward import getReward
    msgs = [
        {"role": "system", "content": "Answer the question or perform the task."},
        {"role": "user", "content": "Summarize Kubernetes in one line."},
        {"role": "assistant", "content": "Kubernetes orchestrates containerized workloads and services."},
    ]
    # Uses DEFAULT_REMOTE_URL = "http://127.0.0.1:8000"
    import asyncio
    score = asyncio.run(getReward(messages=msgs))
    print(score)
"""
from __future__ import annotations

from typing import List, Dict, Optional

import asyncio
import aiohttp
from transformers import AutoTokenizer

# Constants
DEFAULT_REMOTE_URL = "http://127.0.0.1:5001"
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_CLASSIFY_TIMEOUT = 120.0

Message = Dict[str, str]
_tokenizers: dict[str, AutoTokenizer] = {}  # Lightweight cache


async def _fetchFirstRemoteModel(
    session: aiohttp.ClientSession, remoteUrl: str, requestTimeout: float
) -> str:
    """Return the first model id from <remoteUrl>/v1/models (async).

    Args:
        session (aiohttp.ClientSession): Reusable HTTP session.
        remoteUrl (str): Base URL of the remote server.
        requestTimeout (float): HTTP timeout in seconds.

    Returns:
        str: First model identifier.
    """
    url = remoteUrl.rstrip("/") + "/v1/models"
    try:
        timeout = aiohttp.ClientTimeout(total=requestTimeout)
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            payload = await resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch models from {url}: {exc}") from exc

    data = payload.get("data")
    if not data:
        raise RuntimeError("Remote /v1/models returned no models.")

    first = data[0]
    modelId = first.get("id") or first.get("name") if isinstance(first, dict) else str(first)
    if not modelId:
        raise RuntimeError("Could not determine model id from /v1/models response.")
    return modelId


async def _getTokenizer(modelId: str) -> AutoTokenizer:
    """Get or load tokenizer for the given model id (async load).

    Loading the tokenizer may hit disk or network; perform it off the event loop.
    """
    tok = _tokenizers.get(modelId)
    if tok is None:
        tok = await asyncio.to_thread(AutoTokenizer.from_pretrained, modelId)
        _tokenizers[modelId] = tok
    return tok


async def _formatConversation(modelId: str, messages: List[Message]) -> str:
    """Apply model chat template and strip duplicate BOS (async)."""
    tok = await _getTokenizer(modelId)
    formatted = tok.apply_chat_template(messages, tokenize=False)
    bos = getattr(tok, "bos_token", None)
    if bos and isinstance(formatted, str) and formatted.startswith(bos):
        formatted = formatted[len(bos) :]
    return formatted


def _trimAndRescale(x: float, *, cutoff: float = 0.7) -> float:
    """Trim at ``cutoff`` and rescale to [0, 1].

    Args:
        x (float): Original score, typically in [0, 1].
        cutoff (float): Upper trim threshold before rescaling. Defaults to 0.7.

    Returns:
        float: Score after trimming values above ``cutoff`` and rescaling so
            that ``cutoff`` maps to 1.0 and 0 maps to 0.0.

    Raises:
        ValueError: If ``cutoff`` is not positive.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")

    # Clamp to [0, 1] first to be safe, then trim to cutoff and rescale.
    y = max(0.0, min(1.0, float(x)))
    y = min(y, cutoff)
    return y / cutoff


async def _classify(
    session: aiohttp.ClientSession,
    remoteUrl: str,
    modelId: str,
    inputs: List[str],
    classifyTimeout: float,
) -> List[float]:
    """POST formatted inputs to /classify and return scalar scores (async)."""
    url = remoteUrl.rstrip("/") + "/classify"
    body = {"model": modelId, "input": inputs}
    try:
        timeout = aiohttp.ClientTimeout(total=classifyTimeout)
        async with session.post(url, json=body, timeout=timeout) as resp:
            resp.raise_for_status()
            payload = await resp.json()
    except Exception as exc:
        raise RuntimeError(f"Remote classify failed at {url}: {exc}") from exc

    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(inputs):
        raise RuntimeError("Remote classify returned unexpected number of results.")

    scores: List[float] = []
    for item in data:
        if isinstance(item, dict):
            probs = item.get("probs")
            if isinstance(probs, list) and probs:
                raw = float(probs[-1])
                scores.append(_trimAndRescale(raw))
                continue
            if "score" in item:
                try:
                    raw = float(item["score"])
                    scores.append(_trimAndRescale(raw))
                    continue
                except Exception:
                    pass
        raise RuntimeError("Remote classify item missing 'probs' or 'score'.")
    return scores


async def getReward(
    remoteUrl: str = DEFAULT_REMOTE_URL,
    *,
    messages: List[Message],
    modelId: Optional[str] = None,
    requestTimeout: float = DEFAULT_REQUEST_TIMEOUT,
    classifyTimeout: float = DEFAULT_CLASSIFY_TIMEOUT,
) -> float:
    """Return a scalar reward for a single conversation (async).

    Args:
        remoteUrl (str): Base URL of the remote server. Defaults to DEFAULT_REMOTE_URL.
        messages (List[Message]): Conversation as [{'role','content'}, …].
        modelId (Optional[str]): Remote model id. If None, picks the first from /v1/models.
        requestTimeout (float): Timeout for model discovery.
        classifyTimeout (float): Timeout for classification.

    Returns:
        float: Reward score for the conversation.
    """
    async with aiohttp.ClientSession() as session:
        model = modelId or await _fetchFirstRemoteModel(session, remoteUrl, requestTimeout)
        formatted = await _formatConversation(model, messages)
        [score] = await _classify(session, remoteUrl, model, [formatted], classifyTimeout)
        return float(score)


async def getRewards(
    remoteUrl: str = DEFAULT_REMOTE_URL,
    *,
    conversations: List[List[Message]],
    modelId: Optional[str] = None,
    requestTimeout: float = DEFAULT_REQUEST_TIMEOUT,
    classifyTimeout: float = DEFAULT_CLASSIFY_TIMEOUT,
) -> List[float]:
    """Return scalar rewards for a batch of conversations (async).

    Args:
        remoteUrl (str): Base URL of the remote server. Defaults to DEFAULT_REMOTE_URL.
        conversations (List[List[Message]]): Batch of conversations.
        modelId (Optional[str]): Remote model id. If None, picks the first from /v1/models.
        requestTimeout (float): Timeout for model discovery.
        classifyTimeout (float): Timeout for classification.

    Returns:
        List[float]: Reward scores aligned with input order.
    """
    async with aiohttp.ClientSession() as session:
        model = modelId or await _fetchFirstRemoteModel(session, remoteUrl, requestTimeout)
        inputs: List[str] = list(
            await asyncio.gather(*(_formatConversation(model, msgs) for msgs in conversations))
        )
        return await _classify(session, remoteUrl, model, inputs, classifyTimeout)
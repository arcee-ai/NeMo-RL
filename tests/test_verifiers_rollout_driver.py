"""Tests for the verifiers rollout driver utilities."""

from dataclasses import dataclass
from typing import cast

import pytest
import verifiers as vf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rlkit.inference.verifiers_rollout_driver import AppendOnlyTokenizerError, VerifiersRolloutDriver


@dataclass
class _StubEnv:
    message_type: str = "chat"


def _make_tokenizer(chat_template: str) -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "user:": 3,
        "assistant:": 4,
        "hi": 5,
        "ok": 6,
        "1": 7,
        "2": 8,
        "\n": 9,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # pyrefly: ignore[read-only]
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    hf_tok.chat_template = chat_template
    return hf_tok


def _driver_for_tokenizer(tokenizer: PreTrainedTokenizerFast) -> VerifiersRolloutDriver:
    cfg = type("Cfg", (), {})()
    cfg.chat_template = None
    cfg.chat_template_kwargs = {}
    cfg.add_generation_prompt = True
    cfg.continue_final_message = False
    cfg.add_special_tokens = False
    cfg.tool_call_parser = None
    cfg.sampling_args = {}
    env = _StubEnv()
    return VerifiersRolloutDriver(
        cast(vf.Environment, env),
        backend=None,
        tokenizer=tokenizer,
        inference_config=cfg,
        max_seq_len=128,
    )


def test_append_only_chat_template_allows_append() -> None:
    """Allow append-only chat template tokenization."""
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer = _make_tokenizer(chat_template)
    driver = _driver_for_tokenizer(tokenizer)
    state = cast(vf.State, {})

    prompt1 = cast(vf.Messages, [{"role": "user", "content": "hi"}])
    driver._tokenize_prompt(state, prompt1)

    prompt2 = cast(
        vf.Messages,
        [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "hi"},
        ],
    )
    driver._tokenize_prompt(state, prompt2)


def test_append_only_chat_template_rejects_non_append() -> None:
    """Reject chat templates that rewrite previous tokens."""
    chat_template = (
        "{{ messages|length }}\n"
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer = _make_tokenizer(chat_template)
    driver = _driver_for_tokenizer(tokenizer)
    state = cast(vf.State, {})

    prompt1 = cast(vf.Messages, [{"role": "user", "content": "hi"}])
    driver._tokenize_prompt(state, prompt1)

    prompt2 = cast(
        vf.Messages,
        [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        ],
    )
    with pytest.raises(AppendOnlyTokenizerError):
        driver._tokenize_prompt(state, prompt2)


def test_append_only_state_accumulates_tokens() -> None:
    """Persist prompt token ids and extend them on new turns."""
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer = _make_tokenizer(chat_template)
    driver = _driver_for_tokenizer(tokenizer)
    state = cast(vf.State, {})

    prompt1 = cast(vf.Messages, [{"role": "user", "content": "hi"}])
    first = driver._tokenize_prompt(state, prompt1)
    first_ids = list(first.prompt_ids)

    prompt2 = cast(
        vf.Messages,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "hi"},
        ],
    )
    second = driver._tokenize_prompt(state, prompt2)

    assert second.prompt_ids[: len(first_ids)] == first_ids
    assert state["_prompt_token_ids"] == second.prompt_ids
    assert len(second.prompt_ids) > len(first_ids)


def test_vllm_tool_parser_pythonic() -> None:
    """Parse tool calls with vLLM's pythonic parser."""
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer = _make_tokenizer(chat_template)
    cfg = type("Cfg", (), {})()
    cfg.chat_template = None
    cfg.chat_template_kwargs = {}
    cfg.add_generation_prompt = True
    cfg.continue_final_message = False
    cfg.add_special_tokens = False
    cfg.tool_call_parser = "pythonic"
    cfg.sampling_args = {}
    env = _StubEnv()
    driver = VerifiersRolloutDriver(
        cast(vf.Environment, env),
        backend=None,
        tokenizer=tokenizer,
        inference_config=cfg,
        max_seq_len=128,
    )

    prompt = cast(list[vf.ChatMessage], [{"role": "user", "content": "hi"}])
    tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
    content, tool_calls = driver._parse_tool_calls(prompt, "[add(x=1)]", [1, 2, 3], tools)

    assert content is None
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "add"

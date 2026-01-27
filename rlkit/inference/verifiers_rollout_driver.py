"""Rollout driver that uses verifiers environments and a vLLM backend."""
import asyncio
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from transformers import PreTrainedTokenizerBase
from verifiers.utils.async_utils import NullAsyncContext
from verifiers.utils.message_utils import strip_nones_from_content
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager


class AppendOnlyTokenizerError(RuntimeError):
    """Raised when chat template tokenization is not append-only."""


@dataclass
class TokenizedPrompt:
    """Tokenized prompt payload with overlength flag."""

    prompt_ids: list[int]
    overlong_prompt: bool


def _tools_to_dicts(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            out.append(tool)
        else:
            model_dump = getattr(tool, "model_dump", None)
            if model_dump is not None:
                out.append(model_dump())
            else:
                out.append(dict(tool))
    return out


class VerifiersRolloutDriver:
    """Drive verifiers rollouts using a token-based backend."""

    def __init__(
        self,
        env: vf.Environment,
        backend,
        tokenizer: PreTrainedTokenizerBase,
        inference_config,
        max_seq_len: int,
    ):
        """Initialize the driver with an env, backend, and tokenizer."""
        self.env = env
        self.backend = backend
        self.tokenizer = tokenizer
        self.cfg = inference_config
        self.max_seq_len = max_seq_len

    def _apply_chat_template(
        self, messages: list[vf.ChatMessage], tools: list[dict[str, Any]] | None
    ) -> list[int]:
        kwargs = dict(self.cfg.chat_template_kwargs)
        if self.cfg.chat_template is not None:
            kwargs["chat_template"] = self.cfg.chat_template
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=bool(self.cfg.add_generation_prompt),
            continue_final_message=bool(self.cfg.continue_final_message),
            add_special_tokens=bool(self.cfg.add_special_tokens),
            **kwargs,
        )

    def _tokenize_prompt(self, state: vf.State, prompt: vf.Messages) -> TokenizedPrompt:
        if self.env.message_type == "completion":
            assert isinstance(prompt, str)
            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=bool(self.cfg.add_special_tokens)
            )
        else:
            assert isinstance(prompt, list)
            prompt = strip_nones_from_content(prompt)
            tools = _tools_to_dicts(state.get("oai_tools"))
            prompt_ids = self._apply_chat_template(prompt, tools)

        prev_ids = state.get("_prompt_token_ids")
        if prev_ids is not None:
            if prompt_ids[: len(prev_ids)] != prev_ids:
                raise AppendOnlyTokenizerError(
                    "Chat template is not append-only for this tokenizer. "
                    "Prompt tokenization changed across turns."
                )
            prompt_ids = prev_ids + prompt_ids[len(prev_ids) :]

        state["_prompt_token_ids"] = prompt_ids
        return TokenizedPrompt(
            prompt_ids=prompt_ids, overlong_prompt=len(prompt_ids) > self.max_seq_len
        )

    def _prepare_sampling_params(
        self, sampling_args: dict[str, Any], prompt_len: int
    ) -> dict[str, Any]:
        params = dict(sampling_args)
        params.pop("extra_body", None)
        max_tokens = params.pop("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = params.pop("max_tokens", None)
        if self.max_seq_len is not None:
            remaining = max(self.max_seq_len - prompt_len, 0)
            max_tokens = remaining if max_tokens is None else min(max_tokens, remaining)
        params["max_tokens"] = max_tokens

        logprobs = params.get("logprobs")
        if logprobs is True:
            params["logprobs"] = 1
        elif logprobs is False:
            params.pop("logprobs", None)

        if params.get("logprobs") is None:
            params["logprobs"] = 1
        return params

    def _parse_tool_calls(
        self,
        prompt_messages: list[vf.ChatMessage],
        completion_text: str,
        completion_token_ids: list[int],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str | None, list[dict[str, Any]] | None]:
        if not self.cfg.tool_call_parser or not tools:
            return completion_text, None

        parser_cls = ToolParserManager.get_tool_parser(self.cfg.tool_call_parser)
        parser = parser_cls(self.tokenizer)
        request = ChatCompletionRequest(
            model=None,
            messages=prompt_messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            info = parser.extract_tool_calls(
                completion_text, request, token_ids=completion_token_ids
            )
        except TypeError:
            info = parser.extract_tool_calls(completion_text, request)

        if info is None or not info.tools_called:
            return completion_text, None

        tool_calls = []
        for tool_call in info.tool_calls:
            model_dump = getattr(tool_call, "model_dump", None)
            tool_calls.append(model_dump() if model_dump else dict(tool_call))
        return info.content, tool_calls

    def _build_tokens(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        completion_logprobs: list[float],
        finish_reason: str | None,
    ) -> vf.TrajectoryStepTokens:
        prompt_mask = [0] * len(prompt_ids)
        completion_mask = [1] * len(completion_ids)

        overlong_prompt = len(prompt_ids) > self.max_seq_len
        is_truncated = finish_reason == "length"

        if self.max_seq_len is not None and len(prompt_ids) + len(completion_ids) > self.max_seq_len:
            is_truncated = True
            allowed = max(self.max_seq_len - len(prompt_ids), 0)
            completion_ids = completion_ids[:allowed]
            completion_mask = completion_mask[:allowed]
            completion_logprobs = completion_logprobs[:allowed]

        return vf.TrajectoryStepTokens(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            overlong_prompt=overlong_prompt,
            is_truncated=is_truncated,
        )

    async def rollout(
        self,
        input: vf.RolloutInput,
        sampling_args: dict[str, Any],
    ) -> vf.State:
        """Run a single rollout using tokenized prompts and vLLM output."""
        merged_sampling_args = dict(getattr(self.env, "sampling_args", {}) or {})
        merged_sampling_args.update(sampling_args or {})
        state = await self.env.init_state(input, None, "policy", merged_sampling_args)
        try:
            state = await self.env.setup_state(state)
        except vf.Error as e:
            state["error"] = e
        while not await self.env.is_completed(state):
            try:
                prompt_messages = await self.env.get_prompt_messages(state)
                if state.get("final_env_response") is not None:
                    continue

                tokenized = self._tokenize_prompt(state, prompt_messages)
                if tokenized.overlong_prompt:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                    continue

                sampling_params = self._prepare_sampling_params(
                    merged_sampling_args, len(tokenized.prompt_ids)
                )

                result = await self.backend.generate(
                    prompt_token_ids=tokenized.prompt_ids,
                    sampling_params_dict=sampling_params,
                )

                tools = _tools_to_dicts(state.get("oai_tools"))
                content, tool_calls = self._parse_tool_calls(
                    prompt_messages, result.text, result.completion_token_ids, tools
                )

                completion_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if tool_calls is not None:
                    completion_msg["tool_calls"] = tool_calls
                completion_messages = [completion_msg]

                tokens = self._build_tokens(
                    tokenized.prompt_ids,
                    result.completion_token_ids,
                    result.completion_logprobs,
                    result.finish_reason,
                )
                trajectory_step = vf.TrajectoryStep(
                    prompt=prompt_messages,
                    completion=completion_messages,
                    response=None,
                    tokens=tokens,
                    reward=None,
                    advantage=None,
                    is_truncated=tokens["is_truncated"],
                    trajectory_id=state["trajectory_id"],
                    extras={},
                )
                await self.env.add_trajectory_step(state, trajectory_step)
            except AppendOnlyTokenizerError as e:
                state["error"] = e
            except Exception as e:
                state["error"] = e
        await self.env.render_completion(state)
        return state

    async def generate_group(
        self,
        inputs: list[vf.RolloutInput],
        sampling_args: dict[str, Any],
    ) -> list[vf.State]:
        """Generate a group of rollouts and score them if configured."""
        states = await asyncio.gather(
            *[self.rollout(input_item, sampling_args) for input_item in inputs]
        )
        if self.env.score_rollouts:
            await self.env.rubric.score_group(states, score_sem=NullAsyncContext())
        else:
            await self.env.rubric.dummy_score_group(states)
        return list(states)

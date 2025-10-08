# serve_vllm.py
import asyncio
import logging
from typing import Any, Optional
from weakref import WeakValueDictionary

from fastapi import FastAPI, Request
from ray import serve
import torch

# Root FastAPI app used as Serve ingress. We will mount vLLM's app onto this.
_serve_app = FastAPI()


_TOKEN_ID_LOGPROB_PATCH_APPLIED = False
_REQUEST_OUTPUT_REGISTRY: dict[str, Any] | None = None


def _ensure_full_sequence_logprob_patch() -> None:
    global _TOKEN_ID_LOGPROB_PATCH_APPLIED, _REQUEST_OUTPUT_REGISTRY
    if _TOKEN_ID_LOGPROB_PATCH_APPLIED:
        return

    try:
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionLogProbs,
            ChatCompletionLogProbsContent,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.serving_chat import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.protocol import OpenAIBaseModel
        from vllm.outputs import RequestOutput
    except Exception:
        # If vLLM is not available yet, skip patching until it is.
        return

    if _REQUEST_OUTPUT_REGISTRY is None:
        _REQUEST_OUTPUT_REGISTRY = {}

    registry = _REQUEST_OUTPUT_REGISTRY

    # Patch to RequestOutput initializer to log it for later access
    if not getattr(RequestOutput, "_rlkit_full_sequence_logprob_patch", False):
        logging.info("Patching vLLM's RequestOutput.__init__ to register request outputs")
        original_request_output_init = RequestOutput.__init__

        def patched_request_output_init(self, *args, **kwargs):
            original_request_output_init(self, *args, **kwargs)
            registry[self.request_id] = self

        RequestOutput.__init__ = patched_request_output_init  # type: ignore[assignment]
        RequestOutput._rlkit_full_sequence_logprob_patch = True  # type: ignore[attr-defined]

    # Patch to OpenAIServingChat.chat_completion_full_generator to return full sequences
    if not getattr(OpenAIServingChat, "_rlkit_full_sequence_logprob_patch_chat", False):
        logging.info("Patching vLLM's OpenAIServingChat.chat_completion_full_generator to return full sequences")
        original_chat_completion_full_generator = (
            OpenAIServingChat.chat_completion_full_generator
        )

        async def patched_chat_completion_full_generator(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
        ):
            response = await original_chat_completion_full_generator(
                self,
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )

            should_patch = (
                isinstance(response, ChatCompletionResponse)
                and request.return_tokens_as_token_ids
                and request.logprobs is not None
                and not request.echo
            )
            if not should_patch:
                if registry is not None:
                    registry.pop(request_id, None)
                return response

            final_res = registry.get(request_id) if registry is not None else None
            prompt_token_ids = None
            if final_res is not None:
                prompt_ids_attr = getattr(final_res, "prompt_token_ids", None)
                if prompt_ids_attr:
                    prompt_token_ids = list(prompt_ids_attr)
            if not prompt_token_ids:
                raise Exception(
                    "Full-sequence logprob monkey-patch failed: no prompt token ids"
                )

            for choice in response.choices:
                logprobs_payload = choice.logprobs
                if logprobs_payload is None:
                    continue
                
                existing_content = list(logprobs_payload.content or [])
                prompt_content = []
                for token_id in prompt_token_ids:
                    token_str = f"token_id:{token_id}"
                    token_bytes = list(token_str.encode("utf-8", errors="replace"))
                    content = ChatCompletionLogProbsContent(
                        token=token_str,
                        logprob=-9999.0,
                        bytes=token_bytes,
                        top_logprobs=[],
                    )
                    prompt_content.append(content)

                choice.logprobs = ChatCompletionLogProbs(
                    content=prompt_content + existing_content,
                )

            if registry is not None:
                registry.pop(request_id, None)

            return response

        OpenAIServingChat.chat_completion_full_generator = (  # type: ignore[assignment]
            patched_chat_completion_full_generator
        )
        OpenAIServingChat._rlkit_full_sequence_logprob_patch_chat = True  # type: ignore[attr-defined]

    _TOKEN_ID_LOGPROB_PATCH_APPLIED = True


@serve.deployment(max_ongoing_requests=10000)
@serve.ingress(_serve_app)
class VLLMOpenAIServe:
    def __init__(
        self,
        model: str,
        served_model_name: str = "policy",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.7,
        data_parallel_size: int = 1,
        extra_cli_args: Optional[list[str]] = None,
        worker_extension_cls: str = "rlkit.models.generation.vllm_http.worker_ext.VllmHttpWorkerExtension",
        tool_call_parser: str | None = None,
    ):
        for _name in [
            "uvicorn.access",
            "uvicorn.error",
            "ray.serve",
            "ray.serve.deployment",
            "ray.serve.request_summary"
        ]:
            try:
                if _name == "uvicorn.access":
                    logging.getLogger(_name).disabled = True
                else:
                    logging.getLogger(_name).setLevel(logging.ERROR)
            except Exception:
                pass

        args = [
            "--model", model,
            "--served-model-name", served_model_name,
            "--distributed-executor-backend", "ray",
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--pipeline-parallel-size", str(pipeline_parallel_size),
            "--max-model-len", str(max_model_len),
            "--logprobs-mode", "processed_logprobs",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--data-parallel-size", str(data_parallel_size)
        ]
        if tool_call_parser:
            args += ["--tool-call-parser", tool_call_parser]
            args += ["--enable-auto-tool-choice"]
        if extra_cli_args:
            args += extra_cli_args
        
        # We have to import these here so we can import some of these classes in the main process
        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

        parser = FlexibleArgumentParser(description="vLLM OAI app for Ray Serve")
        parser = make_arg_parser(parser)
        self._args = parser.parse_args(args=args)
        validate_parsed_serve_args(self._args)
        
        asyncio.create_task(self._init_app(worker_extension_cls))
    
    async def _init_app(self, worker_extension_cls: str):
        self._worker_extension_cls = worker_extension_cls
        
        # We have to import these here so we can import some of these classes in the main process
        
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import (
            build_app as build_vllm_app,
            build_async_engine_client_from_engine_args,
            init_app_state,
        )
        
        engine_args = AsyncEngineArgs.from_cli_args(self._args)
        engine_args.worker_extension_cls = worker_extension_cls

        _ensure_full_sequence_logprob_patch()

        self._engine_client_ctx = build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing=self._args.disable_frontend_multiprocessing
        )
        self._engine_client = await self._engine_client_ctx.__aenter__()
        
        self._tokenizer = await self._engine_client.get_tokenizer()

        vllm_app = build_vllm_app(self._args)
        
        vllm_config = await self._engine_client.get_vllm_config()
        await init_app_state(self._engine_client, vllm_config, vllm_app.state, self._args)
        vllm_app.state.engine_client = self._engine_client
        
        _serve_app.mount("/", vllm_app)
    
    @_serve_app.get("/sanity_check")
    async def _sanity_check(self):
        return {"status": "ok"}
    
    async def admin_check(self) -> bool:
        return True
    
    async def admin_reset_prefix_cache(self) -> bool:
        await self._engine_client.collective_rpc("reset_prefix_cache", args=tuple())
        return True
    
    async def admin_reset_prefix_cache_async(self) -> bool:
        await self._engine_client.collective_rpc("reset_prefix_cache", args=tuple())
        return True
        
    async def admin_init_collective(self, rank_prefix: int, ip: str, port: int, world_size: int) -> bool:
        # Broadcast same args to all engine workers; extension computes rank from local rank
        await self._engine_client.collective_rpc("init_collective", args=(rank_prefix, ip, port, world_size))
        return True

    async def admin_prepare_refit_info(self, state_dict_info: dict) -> bool:
        await self._engine_client.collective_rpc("prepare_refit_info", args=(state_dict_info,))
        return True

    async def admin_update_from_collective(self) -> bool:
        results = await self._engine_client.collective_rpc("update_weights_from_collective", args=tuple())
        return bool(results and results[0])

    async def admin_report_device_id(self) -> list[str]:
        return await self._engine_client.collective_rpc("report_device_id", args=tuple())

    def maybe_parse_tool_calls(self, parser_name: str | None, texts: list[str]) -> list[dict[str, Any]]:
        """Parse tool calls from generated texts if a parser is configured.

        Returns a list aligned with `texts`, each entry a dict (model_dump) or None.
        """
        
        # For some reason, this file is imported in contexts outside of the vLLM worker.
        # As such, this import needs to be here rather than at the top level.
        from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager, ToolParser
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        try:
            ParserCls = ToolParserManager.get_tool_parser(parser_name)
        except Exception:
            print(f"Failed to get tool parser {parser_name}, returning empty tool calls")
            return [{}] * len(texts)

        if self._tokenizer is None:
            print("No tokenizer found, returning empty tool calls")
            return [{}] * len(texts)

        try:
            parser: ToolParser = ParserCls(self._tokenizer)
            # Dummy request for parser shim.
            req = ChatCompletionRequest(
                messages=[{"role": "user", "content": ""}],
                tool_choice="auto",
                tools=[],
            )
            results: list[dict[str, Any] | None] = []
            for text in texts:
                try:
                    info = parser.extract_tool_calls(text, req)
                    results.append(info.model_dump())
                except Exception as e:
                    print(f"Failed to parse tool calls for text {text}, returning empty tool call: {e}")
                    results.append({})
            
            return results
        except Exception as e:
            print(f"Failed to parse tool calls, returning empty tool calls: {e}")
            return [{}] * len(texts)

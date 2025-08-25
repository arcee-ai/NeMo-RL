# serve_vllm.py
import asyncio
from typing import Any, Optional
from ray import serve
from fastapi import FastAPI, Request
import torch

# Root FastAPI app used as Serve ingress. We will mount vLLM's app onto this.
_serve_app = FastAPI()


@serve.deployment
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
        worker_extension_cls: str = "nemo_rl.models.generation.vllm_http.worker_ext.VllmHttpWorkerExtension",
    ):
        args = [
            "--model", model,
            "--served-model-name", served_model_name,
            "--distributed-executor-backend", "ray",
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--pipeline-parallel-size", str(pipeline_parallel_size),
            "--max-model-len", str(max_model_len),
            "--logprobs-mode", "processed_logprobs",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--data-parallel-size", str(data_parallel_size),
        ]
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

        self._engine_client_ctx = build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing=self._args.disable_frontend_multiprocessing
        )
        self._engine_client = await self._engine_client_ctx.__aenter__()

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

    def maybe_parse_tool_calls(self, texts: list[str]) -> list[dict[str, Any]]:
        """Parse tool calls from generated texts if a parser is configured.

        Returns a list aligned with `texts`, each entry a dict (model_dump) or None.
        """
        
        # For some reason, this file is imported in contexts outside of the vLLM worker.
        # As such, this import needs to be here rather than at the top level.
        from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager, ToolParser
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        parser_name = self.cfg["vllm_cfg"].get("tool_parser", None)
        if parser_name is None:
            return [{}] * len(texts)

        try:
            ParserCls = ToolParserManager.get_tool_parser(parser_name)
        except Exception:
            return [{}] * len(texts)

        try:
            tokenizer = self.llm.get_tokenizer()
        except Exception:
            tokenizer = None

        if tokenizer is None:
            return [{}] * len(texts)

        try:
            parser: ToolParser = ParserCls(tokenizer)
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
                except Exception:
                    results.append({})
            
            return results
        except Exception:
            return [{}] * len(texts)
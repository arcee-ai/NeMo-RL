# serve_vllm.py
import asyncio
from typing import Optional
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import (
    build_app as build_vllm_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
)
from vllm.utils import FlexibleArgumentParser
from fastapi import FastAPI, Request

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
        max_model_len: int = 8192,
        extra_cli_args: Optional[list[str]] = None,
        worker_extension_cls: str = "nemo_rl.models.generation.vllm_http.worker_ext.CheckpointWorker",
    ):
        args = [
            "--model", model,
            "--served-model-name", served_model_name,
            "--distributed-executor-backend", "ray",
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
            "--logprobs-mode", "processed_logprobs",
        ]
        if extra_cli_args:
            args += extra_cli_args

        parser = FlexibleArgumentParser(description="vLLM OAI app for Ray Serve")
        parser = make_arg_parser(parser)
        self._args = parser.parse_args(args=args)
        validate_parsed_serve_args(self._args)
        
        asyncio.create_task(self._init_app(worker_extension_cls))
    
    async def _init_app(self, worker_extension_cls: str):
        self._worker_extension_cls = worker_extension_cls
        
        engine_args = AsyncEngineArgs.from_cli_args(self._args)
        # engine_args.worker_extension_cls = worker_extension_cls

        self._engine_client_ctx = build_async_engine_client_from_engine_args(
            engine_args, self._args.disable_frontend_multiprocessing, None
        )
        self._engine_client = await self._engine_client_ctx.__aenter__()

        vllm_app = build_vllm_app(self._args)
        
        _serve_app.mount("/", vllm_app)

    @_serve_app.get("/sanity_check")
    async def _sanity_check(self):
        return {"status": "ok"}
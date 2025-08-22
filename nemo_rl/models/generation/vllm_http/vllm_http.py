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

        self._worker_extension_cls = worker_extension_cls
        self._engine_client_ctx = None
        self._engine_client = None

        new_app = FastAPI()
        
        @new_app.get("/sanity_check_2")
        async def _sanity_check_2(self):
            return {"status": "great success"}
        
        _serve_app.mount("/v1", new_app)

    @_serve_app.get("/sanity_check")
    async def _sanity_check(self):
        return {"status": "ok"}
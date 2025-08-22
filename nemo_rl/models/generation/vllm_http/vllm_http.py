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

        # Build vLLM FastAPI app and mount it at root so its /v1/* routes are preserved.
        vllm_app = build_vllm_app(self._args)
        
        # Mount the vLLM app as the root app served by Serve.
        # _serve_app.mount("/", vllm_app)

    @_serve_app.get("/sanity_check")
    async def _sanity_check():
        return {"status": "ok"}

    # Expose weight management endpoints on the vLLM app itself under /v1/*
    @_serve_app.post("/v1/update_weights")
    async def _update_weights(request: Request):
        data = await request.json()
        model_path = data.get("model_path")
        engine_client = request.app.state.engine_client
        await engine_client.collective_rpc("update_weights", args=(model_path,))
        return {"status": "ok"}
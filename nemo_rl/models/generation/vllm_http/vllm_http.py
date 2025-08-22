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


@serve.deployment()
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

        # Build vLLM FastAPI app and include its routes under the ingress app.
        # Using include_router avoids path prefix confusion when Serve strips route_prefix.
        vllm_app = build_vllm_app(self._args)
        _serve_app.include_router(vllm_app.router)

        # Expose weight management endpoints on the ingress app.
        @_serve_app.post("/v1/update_weights")
        async def _update_weights(request: Request):
            data = await request.json()
            model_path = data.get("model_path")
            engine_client = request.app.state.engine_client
            await engine_client.collective_rpc("update_weights", args=(model_path,))
            return {"status": "ok"}

        @_serve_app.post("/v1/reload_weights")
        async def _reload_weights(request: Request):
            engine_client = request.app.state.engine_client
            await engine_client.collective_rpc("reload_weights")
            return {"status": "ok"}

        # Register startup/shutdown events to init/teardown engine client lazily via ASGI lifespan.
        @_serve_app.on_event("startup")
        async def _startup():
            engine_args = AsyncEngineArgs.from_cli_args(self._args)
            engine_args.worker_extension_cls = self._worker_extension_cls
            self._engine_client_ctx = build_async_engine_client_from_engine_args(
                engine_args, self._args.disable_frontend_multiprocessing, None
            )
            self._engine_client = await self._engine_client_ctx.__aenter__()
            vllm_config = await self._engine_client.get_vllm_config()
            await init_app_state(self._engine_client, vllm_config, _serve_app.state, self._args)
            # Provide direct access for custom endpoints
            _serve_app.state.engine_client = self._engine_client

        @_serve_app.on_event("shutdown")
        async def _shutdown():
            if self._engine_client_ctx is not None:
                await self._engine_client_ctx.__aexit__(None, None, None)
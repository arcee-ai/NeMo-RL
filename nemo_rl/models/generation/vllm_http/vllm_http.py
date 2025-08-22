# serve_vllm.py
import asyncio
from typing import Optional
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
)
from vllm.utils import FlexibleArgumentParser

@serve.deployment()
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

        # Defer async initialization to first request to avoid nesting event loops.
        self._worker_extension_cls = worker_extension_cls
        self._asgi = None
        self._init_task = None
        self._engine_client_ctx = None
        self._engine_client = None

    async def _init_app(self, worker_extension_cls: str):
        if self._asgi is not None:
            return
        engine_args = AsyncEngineArgs.from_cli_args(self._args)
        engine_args.worker_extension_cls = worker_extension_cls

        self._engine_client_ctx = build_async_engine_client_from_engine_args(
            engine_args, self._args.disable_frontend_multiprocessing, None
        )
        self._engine_client = await self._engine_client_ctx.__aenter__()

        app = build_app(self._args)

        @app.post("/update_weights")
        async def _update_weights(request):
            data = await request.json()
            model_path = data.get("model_path")
            await self._engine_client.collective_rpc("update_weights", args=(model_path,))
            return {"status": "ok"}

        @app.post("/reload_weights")
        async def _reload_weights(_):
            await self._engine_client.collective_rpc("reload_weights")
            return {"status": "ok"}

        vllm_config = await self._engine_client.get_vllm_config()
        await init_app_state(self._engine_client, vllm_config, app.state, self._args)

        self._asgi = app

    async def __call__(self, scope, receive, send):
        if self._asgi is None:
            if self._init_task is None:
                self._init_task = asyncio.create_task(self._init_app(self._worker_extension_cls))
            await self._init_task
        await self._asgi(scope, receive, send)
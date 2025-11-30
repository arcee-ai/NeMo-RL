# serve_vllm.py
import asyncio
import logging
from typing import Any, Optional

from fastapi import FastAPI
import ray
import uvicorn

# Root FastAPI app used as Serve ingress. We will mount vLLM's app onto this.
_serve_app = FastAPI()

@ray.remote(num_cpus=1)
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
        worker_extension_cls: str = "rlkit.models.generation.worker_ext.VllmHttpWorkerExtension",
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
            "--data-parallel-size", str(data_parallel_size),
            "--trust-remote-code",
            # Performance optimizations
            "--max-num-seqs", "2048",  # Allow more concurrent sequences for better batching
            "--enable-chunked-prefill",  # Prevent long prefills from blocking decode
        ]
        if tool_call_parser:
            args += ["--tool-call-parser", tool_call_parser]
            args += ["--enable-auto-tool-choice"]
        if extra_cli_args:
            args += extra_cli_args
        
        # We have to import these here so we can import some of these classes in the main process
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

        parser = FlexibleArgumentParser(description="vLLM OAI app for Ray Serve")
        parser = make_arg_parser(parser)
        self._args = parser.parse_args(args=args)
        validate_parsed_serve_args(self._args)
        
        # Track engine initialization status
        self._engine_initialized = False
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
        
        # Mark engine as initialized
        self._engine_initialized = True
        
        self._tokenizer = await self._engine_client.get_tokenizer()

        vllm_app = build_vllm_app(self._args)
        
        await init_app_state(self._engine_client, vllm_app.state, self._args)
        vllm_app.state.engine_client = self._engine_client
        
        _serve_app.mount("/", vllm_app)
        
        config = uvicorn.Config(
            _serve_app,
            host="0.0.0.0",
            port=8000,
            limit_concurrency=None,  # No limit on concurrent connections
            limit_max_requests=None,  # No limit on total requests
            backlog=8192,  # Increase connection backlog
            timeout_keep_alive=120,  # Keep connections alive longer
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    @_serve_app.get("/sanity_check")
    async def _sanity_check(self):
        return {"status": "ok"}
    
    async def admin_check(self) -> bool:
        return True
    
    async def admin_engine_ready(self) -> bool:
        """Check if vLLM engine has finished initializing."""
        return getattr(self, '_engine_initialized', False)
    
    async def admin_get_ip(self) -> str:
        return ray.util.get_node_ip_address()
    
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

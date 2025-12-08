"""Single-node vLLM HTTP server."""
from typing import Literal
import asyncio
from typing import Optional

from fastapi import FastAPI
import ray
import uvicorn


@ray.remote(num_cpus=1)
class VLLMOpenAIServe:
    """Single-node vLLM HTTP server."""
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
        worker_extension_cls: str = "rlkit.inference.worker_ext.VllmHttpWorkerExtension",
        tool_call_parser: str | None = None,
    ):
        """Initialize a vLLM HTTP server on one node."""
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

        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

        parser = FlexibleArgumentParser(description="vLLM OAI app for Ray Serve")
        parser = make_arg_parser(parser)
        _args = parser.parse_args(args=args)
        assert _args is not None, "Failed to parse arguments"
        self._args = _args
        validate_parsed_serve_args(self._args)

        # Track engine initialization status
        self._engine_initialized = False
        self.create_task = asyncio.create_task(self._init_app(worker_extension_cls))

    async def _init_app(self, worker_extension_cls: str):
        """Async task to start vLLM HTTP app."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import (
            build_app as build_vllm_app,
            build_async_engine_client_from_engine_args,
            init_app_state,
        )

        self._worker_extension_cls = worker_extension_cls

        engine_args = AsyncEngineArgs.from_cli_args(self._args)
        engine_args.worker_extension_cls = worker_extension_cls

        self._engine_client_ctx = build_async_engine_client_from_engine_args(
            engine_args, # type: ignore[arg-type] - appears to be type hinting issue in vLLM
            disable_frontend_multiprocessing=self._args.disable_frontend_multiprocessing
        )
        self._engine_client = await self._engine_client_ctx.__aenter__()

        # Mark engine as initialized
        self._engine_initialized = True

        self._tokenizer = await self._engine_client.get_tokenizer()

        vllm_app = build_vllm_app(self._args)

        await init_app_state(self._engine_client, vllm_app.state, self._args)
        vllm_app.state.engine_client = self._engine_client

        # Create the root FastAPI app inside the actor (not at module level)
        # to avoid serialization issues with Ray
        serve_app = FastAPI()
        serve_app.mount("/", vllm_app)

        config = uvicorn.Config(
            serve_app,
            host="0.0.0.0",
            port=8000,
            limit_concurrency=None,  # No limit on concurrent connections
            limit_max_requests=None,  # No limit on total requests
            backlog=8192,  # Increase connection backlog
            timeout_keep_alive=120,  # Keep connections alive longer
            access_log=False,  # Suppress access log spam
            log_level="warning",  # Only show warnings and errors
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def admin_engine_ready(self) -> bool:
        """Check if vLLM engine has finished initializing."""
        return getattr(self, '_engine_initialized', False)

    async def admin_get_ip(self) -> str:
        """Get the IP address of this node."""
        return ray.util.get_node_ip_address()

    async def admin_reset_prefix_cache(self) -> bool:
        """Reset prefix cache on all vLLM workers."""
        await self._engine_client.collective_rpc("reset_prefix_cache", args=tuple())
        return True

    async def admin_reset_prefix_cache_async(self) -> bool:
        """Reset prefix cache on all vLLM workers asynchronously."""
        await self._engine_client.collective_rpc("reset_prefix_cache", args=tuple())
        return True

    async def admin_init_collective(self, rank_prefix: int, ip: str, port: int, world_size: int) -> Literal[True]:
        """Initialize collective communication between vLLM workers and the training rank0 for weight refits.

        Args:
            rank_prefix: The prefix to add to our local rank to get the global rank.
            ip: The IP address of the head node.
            port: The port to use for collective communication.
            world_size: The total number of workers in the collective (train rank0 + inference workers).
        """
        await self._engine_client.collective_rpc("init_collective", args=(rank_prefix, ip, port, world_size))
        return True

    async def admin_prepare_refit_info(self, state_dict_info: dict) -> bool:
        """Prepare for weight refits."""
        await self._engine_client.collective_rpc("prepare_refit_info", args=(state_dict_info,))
        return True

    async def admin_update_from_collective(self) -> bool:
        """Receive weight updates from training workers."""
        results = await self._engine_client.collective_rpc("update_weights_from_collective", args=tuple())
        return bool(results and results[0])

    async def admin_report_device_id(self) -> list[str]:
        """Report device IDs for vLLM workers."""
        return await self._engine_client.collective_rpc("report_device_id", args=tuple())

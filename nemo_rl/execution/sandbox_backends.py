"""Sandbox backend implementations for Beam, E2B, and Docker providers."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
import ray

from nemo_rl.execution.docker_sandbox_pool import DockerSandboxPool


logger = logging.getLogger(__name__)


@dataclass
class SandboxSession:
    """Represents a live sandbox session that must be released."""

    base_url: str
    provider: str
    metadata: Dict[str, Any]
    _release_cb: Callable[[bool], Awaitable[None]]

    async def release(self, healthy: bool = True) -> None:
        await self._release_cb(healthy)


class BaseSandboxBackend:
    """Common interface for sandbox backends."""

    async def acquire(self) -> SandboxSession:  # pragma: no cover - abstract
        raise NotImplementedError

    async def shutdown(self) -> None:  # pragma: no cover - optional hook
        return None


class DockerSandboxBackend(BaseSandboxBackend):
    """Wrapper around the Docker sandbox pool actor."""

    def __init__(
        self,
        image: str,
        warm_pool_size: int,
        max_total_size: int,
        readiness_timeout: float,
        container_port: int,
    ) -> None:
        self.image = image
        self.warm_pool_size = warm_pool_size
        self.max_total_size = max_total_size
        self.readiness_timeout = readiness_timeout
        self.container_port = container_port

        self._pool: Optional[ray.actor.ActorHandle] = None

    async def _ensure_pool(self) -> ray.actor.ActorHandle:
        if self._pool is None:
            self._pool = (
                DockerSandboxPool.options(name=None, lifetime="detached")
                .remote(
                    image=self.image,
                    warm_pool_size=self.warm_pool_size,
                    max_total_size=self.max_total_size,
                    readiness_timeout=self.readiness_timeout,
                    container_port=self.container_port,
                )
            )
            logger.info("Created Docker sandbox pool for image %s", self.image)
        return self._pool

    async def acquire(self) -> SandboxSession:
        pool = await self._ensure_pool()
        lease_ref = pool.acquire.remote()
        lease = await lease_ref

        async def _release(healthy: bool) -> None:
            await pool.release.remote(lease["container_id"], healthy=healthy)

        metadata = {
            "container_id": lease["container_id"],
            "volume_name": lease.get("volume_name"),
        }
        return SandboxSession(
            base_url=lease["base_url"],
            provider="docker",
            metadata=metadata,
            _release_cb=_release,
        )

    async def shutdown(self) -> None:
        if self._pool is None:
            return
        try:
            await self._pool.shutdown.remote()
        finally:
            self._pool = None


class BeamSandboxBackend(BaseSandboxBackend):
    """Sandbox provider that leverages Beam cloud sandboxes."""

    def __init__(
        self,
        template: Optional[str],
        start_command: Optional[str],
        readiness_path: str,
        readiness_timeout: float,
        container_port: int,
        working_dir: str,
        environment: Optional[Dict[str, str]],
        insecure_skip_tls_verify: bool = False,
    ) -> None:
        if not template:
            raise ValueError("BeamSandboxBackend requires a 'template' identifier")

        self.template = template
        self.start_command = start_command
        self.readiness_path = readiness_path or "/doc"
        self.readiness_timeout = readiness_timeout
        self.container_port = container_port
        self.working_dir = working_dir
        self.environment = environment or {}
        self.insecure_skip_tls_verify = insecure_skip_tls_verify

    async def acquire(self) -> SandboxSession:
        sandbox = await asyncio.to_thread(self._create_sandbox)

        try:
            public_url = await asyncio.to_thread(self._ensure_service, sandbox)
            await self._wait_for_readiness(public_url)
        except Exception:
            await asyncio.to_thread(self._cleanup_sandbox, sandbox)
            raise

        sandbox_id = getattr(sandbox, "id", str(uuid.uuid4()))

        async def _release(healthy: bool) -> None:
            # Beam sandboxes are always torn down after use for isolation.
            await asyncio.to_thread(self._cleanup_sandbox, sandbox)

        metadata = {"sandbox_id": sandbox_id}
        return SandboxSession(
            base_url=public_url.rstrip("/"),
            provider="beam",
            metadata=metadata,
            _release_cb=_release,
        )

    # ------------------------------------------------------------------
    # Beam helpers
    # ------------------------------------------------------------------
    def _create_sandbox(self):
        beam_module = importlib.import_module("beam")
        
        # Use Image.from_registry for Docker images or templates
        image_cls = getattr(beam_module, "Image")
        sandbox_cls = getattr(beam_module, "Sandbox")
        
        image = image_cls.from_registry(self.template)
        sandbox_cfg = sandbox_cls(image=image)
        
        if hasattr(sandbox_cfg, "create"):
            sandbox = sandbox_cfg.create()
        else:
            sandbox = sandbox_cfg
        return sandbox

    def _ensure_service(self, sandbox) -> str:
        if self.start_command:
            command_line = ["bash", "-lc", self.start_command]
            process_iface = getattr(sandbox, "process", None)
            if process_iface and hasattr(process_iface, "exec"):
                exec_fn = process_iface.exec
                # Execute command - long-running processes will run in background automatically
                exec_fn(*command_line, cwd=self.working_dir, env=self.environment)

        try:
            url = sandbox.expose_port(self.container_port)
        except TypeError:
            # Older SDKs expose synchronous API without kwargs
            url = sandbox.expose_port(self.container_port)
        return url

    def _cleanup_sandbox(self, sandbox) -> None:
        # Try termination methods in order of preference based on Beam API
        for attr in ("terminate", "delete", "close", "stop", "kill"):
            method = getattr(sandbox, attr, None)
            if callable(method):
                try:
                    method()
                    return
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    sandbox_id = getattr(sandbox, "sandbox_id", lambda: "unknown")
                    if callable(sandbox_id):
                        sandbox_id = sandbox_id()
                    logger.debug("Beam sandbox %s cleanup via %s failed: %s", sandbox_id, attr, exc)

    async def _wait_for_readiness(self, base_url: str) -> None:
        deadline = time.time() + self.readiness_timeout
        path = self.readiness_path

        if not path.startswith("/"):
            path = "/" + path

        async with httpx.AsyncClient(verify=not self.insecure_skip_tls_verify) as client:
            while True:
                try:
                    response = await client.get(f"{base_url}{path}")
                    if response.status_code < 500:
                        return
                except Exception as exc:
                    logger.debug("Beam readiness check failed: %s", exc)

                if time.time() > deadline:
                    raise TimeoutError(f"Beam sandbox at {base_url} did not become ready within {self.readiness_timeout}s")

                await asyncio.sleep(1.0)


class E2BSandboxBackend(BaseSandboxBackend):
    """Sandbox provider backed by E2B sandboxes."""

    def __init__(
        self,
        template: str,
        api_key: Optional[str],
        readiness_path: str,
        readiness_timeout: float,
        container_port: int,
        start_command: Optional[str],
        working_dir: str,
        environment: Optional[Dict[str, str]],
        insecure_skip_tls_verify: bool = False,
    ) -> None:
        if not template:
            raise ValueError("E2B sandbox backend requires a template identifier")

        self.template = template
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY must be provided via config or environment")

        self.readiness_path = readiness_path or "/doc"
        self.readiness_timeout = readiness_timeout
        self.container_port = container_port
        self.start_command = start_command
        self.working_dir = working_dir
        self.environment = environment or {}
        self.insecure_skip_tls_verify = insecure_skip_tls_verify

    async def acquire(self) -> SandboxSession:
        sandbox = await self._create_sandbox()

        try:
            base_url = await self._ensure_service(sandbox)
            await self._wait_for_readiness(base_url)
        except Exception:
            await self._cleanup_sandbox(sandbox)
            raise

        sandbox_id = getattr(sandbox, "id", str(uuid.uuid4()))

        async def _release(healthy: bool) -> None:
            await self._cleanup_sandbox(sandbox)

        metadata = {"sandbox_id": sandbox_id}
        return SandboxSession(
            base_url=base_url.rstrip("/"),
            provider="e2b",
            metadata=metadata,
            _release_cb=_release,
        )

    async def _create_sandbox(self):
        module = importlib.import_module("e2b_code_interpreter")
        if hasattr(module, "AsyncSandbox"):
            sandbox = await module.AsyncSandbox.create(template=self.template, api_key=self.api_key)
        else:
            sandbox_cls = getattr(module, "Sandbox")
            sandbox = sandbox_cls(self.template, api_key=self.api_key)
        return sandbox

    async def _ensure_service(self, sandbox) -> str:
        if self.start_command:
            command_args = ["bash", "-lc", self.start_command]
            exec_coro = getattr(sandbox, "process", None)
            if exec_coro and hasattr(exec_coro, "exec"):
                try:
                    await exec_coro.exec(
                        *command_args,
                        cwd=self.working_dir,
                        env=self.environment,
                        background=True,
                    )
                except TypeError:
                    await exec_coro.exec(
                        *command_args,
                        cwd=self.working_dir,
                        env=self.environment,
                    )
            elif hasattr(sandbox, "commands"):
                await sandbox.commands.run(self.start_command)

        get_host = getattr(sandbox, "get_host", None)
        if callable(get_host):
            host = await get_host(self.container_port) if asyncio.iscoroutinefunction(get_host) else get_host(self.container_port)
            if host.startswith("http"):
                return host
            return f"https://{host}"

        # Fallback to expose_port equivalent if available
        expose_port = getattr(sandbox, "expose_port", None)
        if callable(expose_port):
            result = await expose_port(self.container_port) if asyncio.iscoroutinefunction(expose_port) else expose_port(self.container_port)
            return result

        raise RuntimeError("E2B sandbox does not expose get_host or expose_port APIs")

    async def _wait_for_readiness(self, base_url: str) -> None:
        deadline = time.time() + self.readiness_timeout
        path = self.readiness_path
        if not path.startswith("/"):
            path = "/" + path

        async with httpx.AsyncClient(verify=not self.insecure_skip_tls_verify) as client:
            while True:
                try:
                    response = await client.get(f"{base_url}{path}")
                    if response.status_code < 500:
                        return
                except Exception as exc:
                    logger.debug("E2B readiness check failed: %s", exc)

                if time.time() > deadline:
                    raise TimeoutError(f"E2B sandbox at {base_url} did not become ready within {self.readiness_timeout}s")

                await asyncio.sleep(1.0)

    async def _cleanup_sandbox(self, sandbox) -> None:
        for attr in ("kill", "close", "stop", "delete"):
            method = getattr(sandbox, attr, None)
            if callable(method):
                try:
                    await method() if asyncio.iscoroutinefunction(method) else method()
                    return
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.debug("E2B sandbox cleanup via %s failed: %s", attr, exc)


def build_backend_sequence(
    backend_priority: list[str],
    *,
    docker_config: Dict[str, Any],
    beam_config: Optional[Dict[str, Any]] = None,
    e2b_config: Optional[Dict[str, Any]] = None,
) -> list[tuple[str, BaseSandboxBackend]]:
    """Create backend instances following the provided priority list."""

    normalized_priority = [backend.lower() for backend in backend_priority]
    sequence: list[tuple[str, BaseSandboxBackend]] = []

    for backend in normalized_priority:
        if backend == "beam" and beam_config:
            try:
                sequence.append(
                    (
                        "beam",
                        BeamSandboxBackend(
                            template=beam_config.get("template"),
                            start_command=beam_config.get("start_command"),
                            readiness_path=beam_config.get("readiness_path", "/doc"),
                            readiness_timeout=beam_config.get("readiness_timeout", 180.0),
                            container_port=beam_config.get("container_port", 4096),
                            working_dir=beam_config.get("working_dir", "/workspace"),
                            environment=beam_config.get("environment"),
                            insecure_skip_tls_verify=beam_config.get("skip_tls_verify", False),
                        ),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping Beam backend due to configuration error: %s", exc)
        elif backend == "e2b" and e2b_config:
            try:
                sequence.append(
                    (
                        "e2b",
                        E2BSandboxBackend(
                            template=e2b_config.get("template"),
                            api_key=e2b_config.get("api_key"),
                            readiness_path=e2b_config.get("readiness_path", "/doc"),
                            readiness_timeout=e2b_config.get("readiness_timeout", 180.0),
                            container_port=e2b_config.get("container_port", 4096),
                            start_command=e2b_config.get("start_command"),
                            working_dir=e2b_config.get("working_dir", "/workspace"),
                            environment=e2b_config.get("environment"),
                            insecure_skip_tls_verify=e2b_config.get("skip_tls_verify", False),
                        ),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping E2B backend due to configuration error: %s", exc)
        elif backend == "docker":
            sequence.append(
                (
                    "docker",
                    DockerSandboxBackend(
                        image=docker_config["image"],
                        warm_pool_size=docker_config["warm_pool_size"],
                        max_total_size=docker_config["max_total_size"],
                        readiness_timeout=docker_config["readiness_timeout"],
                        container_port=docker_config["container_port"],
                    ),
                )
            )

    if not sequence:
        raise ValueError("No valid sandbox backend configuration found")

    return sequence

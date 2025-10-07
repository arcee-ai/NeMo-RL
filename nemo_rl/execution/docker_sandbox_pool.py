"""Ray-native Docker sandbox pool for isolated code execution environments."""

import asyncio
import logging
import uuid
from typing import Optional, TypedDict

import docker
import httpx
import ray


logger = logging.getLogger(__name__)


class SandboxLease(TypedDict):
    """Represents an exclusive lease on a sandbox container."""
    container_id: str
    base_url: str
    volume_name: str


@ray.remote(num_cpus=0)
class DockerSandboxPool:
    """Ray actor that manages a warm pool of Docker sandbox containers.
    
    Each sandbox runs OpenCode server and provides an isolated execution environment
    for code execution, testing, and evaluation. Containers are:
    - Bound to random localhost ports for security
    - Backed by unique Docker volumes for /workspace isolation
    - Automatically recycled (destroyed and recreated) after use
    - Warmed on initialization and scaled on demand
    
    The pool uses asyncio.Queue for lease management and respects
    a maximum total capacity to prevent resource exhaustion.
    
    Args:
        image: Docker image name (e.g., "ghcr.io/your-org/opencode-worker:latest")
        warm_pool_size: Number of containers to keep warm and ready
        max_total_size: Maximum total number of containers (including in-use)
        readiness_timeout: Seconds to wait for each container to become ready
        container_port: Internal container port for OpenCode server
    
    Example:
        >>> pool = DockerSandboxPool.remote(
        ...     image="opencode-worker:latest",
        ...     warm_pool_size=8,
        ...     max_total_size=32,
        ... )
        >>> lease = await pool.acquire.remote()
        >>> lease = await ray.get(lease)
        >>> # Use lease["base_url"] to interact with sandbox
        >>> await pool.release.remote(lease["container_id"], healthy=True)
    """

    def __init__(
        self,
        image: str,
        warm_pool_size: int = 8,
        max_total_size: int = 32,
        readiness_timeout: float = 120.0,
        container_port: int = 4096,
    ):
        self.image = image
        self.warm_pool_size = warm_pool_size
        self.max_total_size = max_total_size
        self.readiness_timeout = readiness_timeout
        self.container_port = container_port

        self.client = docker.from_env()
        self._ready_queue: asyncio.Queue[SandboxLease] = asyncio.Queue()
        self._in_use: set[str] = set()
        self._init_lock = asyncio.Lock()
        self._scale_lock = asyncio.Lock()
        self._initialized = False

        logger.info(
            f"Initialized DockerSandboxPool: image={image}, "
            f"warm={warm_pool_size}, max={max_total_size}"
        )

    async def _lazy_init(self) -> None:
        """Initialize the warm pool on first use."""
        async with self._init_lock:
            if self._initialized:
                return
            
            logger.info(f"Warming pool with {self.warm_pool_size} containers...")
            spawn_tasks = [self._spawn_ready() for _ in range(self.warm_pool_size)]
            results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Pool initialized: {successful}/{self.warm_pool_size} containers ready")
            
            self._initialized = True

    async def acquire(self, timeout: float = 300.0) -> SandboxLease:
        """Acquire an exclusive lease on a sandbox container.
        
        This method will:
        1. Wait for a ready container from the pool (up to timeout)
        2. If pool is empty and capacity allows, spawn additional containers
        3. Mark the container as in-use
        4. Return a lease with connection details
        
        Args:
            timeout: Maximum seconds to wait for a container
            
        Returns:
            SandboxLease with container_id, base_url, and volume_name
            
        Raises:
            asyncio.TimeoutError: If no container becomes available within timeout
        """
        await self._lazy_init()

        try:
            lease = await asyncio.wait_for(
                self._ready_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Try to scale up once if we haven't hit max capacity
            async with self._scale_lock:
                current_total = self._ready_queue.qsize() + len(self._in_use)
                available_capacity = self.max_total_size - current_total
                
                if available_capacity > 0:
                    to_spawn = min(4, available_capacity)
                    logger.warning(
                        f"Pool exhausted, spawning {to_spawn} additional containers "
                        f"(capacity: {current_total}/{self.max_total_size})"
                    )
                    spawn_tasks = [self._spawn_ready() for _ in range(to_spawn)]
                    await asyncio.gather(*spawn_tasks, return_exceptions=True)
            
            # Retry once after scaling
            lease = await asyncio.wait_for(
                self._ready_queue.get(),
                timeout=timeout,
            )

        self._in_use.add(lease["container_id"])
        logger.debug(
            f"Acquired sandbox: {lease['container_id'][:12]} "
            f"(in-use: {len(self._in_use)}, ready: {self._ready_queue.qsize()})"
        )
        return lease

    async def release(self, container_id: str, healthy: bool = True) -> None:
        """Release a sandbox container back to the pool.
        
        Containers are always destroyed and replaced with fresh ones to ensure
        complete isolation between tasks and prevent state leakage.
        
        Args:
            container_id: Container ID from the lease
            healthy: Whether the container should be replaced (True) or just removed (False)
        """
        self._in_use.discard(container_id)

        try:
            container = self.client.containers.get(container_id)
        except docker.errors.NotFound:
            logger.warning(f"Container {container_id[:12]} not found during release")
            if healthy:
                await self._spawn_ready()
            return

        # Extract volume name before destroying container
        volume_name = None
        try:
            mounts = container.attrs.get("Mounts", [])
            if mounts:
                volume_name = mounts[0].get("Name")
        except Exception as e:
            logger.warning(f"Failed to extract volume name: {e}")

        # Destroy container
        try:
            container.remove(force=True)
            logger.debug(f"Destroyed container: {container_id[:12]}")
        except Exception as e:
            logger.error(f"Failed to remove container {container_id[:12]}: {e}")

        # Destroy volume
        if volume_name:
            try:
                volume = self.client.volumes.get(volume_name)
                volume.remove(force=True)
                logger.debug(f"Destroyed volume: {volume_name}")
            except Exception as e:
                logger.warning(f"Failed to remove volume {volume_name}: {e}")

        # Spawn replacement if healthy
        if healthy:
            await self._spawn_ready()
            logger.debug(
                f"Released and replaced sandbox: {container_id[:12]} "
                f"(in-use: {len(self._in_use)}, ready: {self._ready_queue.qsize()})"
            )

    async def _spawn_ready(self) -> None:
        """Spawn a new container and wait for it to become ready."""
        volume_name = f"sandbox-{uuid.uuid4()}"
        container_name = f"sandbox-{uuid.uuid4()}"

        try:
            # Create dedicated volume
            volume = self.client.volumes.create(name=volume_name)
            logger.debug(f"Created volume: {volume_name}")

            # Start container with random host port binding
            container = self.client.containers.run(
                self.image,
                detach=True,
                remove=False,
                name=container_name,
                ports={f"{self.container_port}/tcp": ("127.0.0.1",)},
                volumes={volume_name: {"bind": "/workspace", "mode": "rw"}},
                # Security: run as non-root user
                user="sandbox",
                # Resource limits to prevent runaway containers
                mem_limit="2g",
                memswap_limit="2g",
                cpu_quota=100000,  # 1 CPU
            )
            logger.debug(f"Started container: {container_name}")

            # Wait for container to become ready
            lease = await self._wait_for_readiness(container, volume_name)
            await self._ready_queue.put(lease)
            
            logger.debug(
                f"Sandbox ready: {container.id[:12]} -> {lease['base_url']} "
                f"(ready: {self._ready_queue.qsize()})"
            )

        except Exception as e:
            logger.error(f"Failed to spawn sandbox: {e}")
            # Cleanup on failure
            try:
                if "container" in locals():
                    self.client.containers.get(container.id).remove(force=True)
            except Exception:
                pass
            try:
                if "volume" in locals():
                    self.client.volumes.get(volume_name).remove(force=True)
            except Exception:
                pass
            raise

    async def _wait_for_readiness(
        self,
        container: docker.models.containers.Container,
        volume_name: str,
    ) -> SandboxLease:
        """Wait for a container to become ready and return its lease.
        
        Readiness is determined by:
        1. Container is running
        2. Port mapping is established
        3. OpenCode server responds to health check
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.readiness_timeout:
                raise TimeoutError(
                    f"Container {container.id[:12]} did not become ready "
                    f"within {self.readiness_timeout}s"
                )

            # Refresh container state
            try:
                container.reload()
            except docker.errors.NotFound:
                raise RuntimeError(f"Container {container.id[:12]} disappeared during startup")

            # Check if container died
            if container.status != "running":
                logs = container.logs().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Container {container.id[:12]} failed to start. "
                    f"Status: {container.status}. Logs:\n{logs}"
                )

            # Check port mapping
            port_info = container.attrs["NetworkSettings"]["Ports"].get(
                f"{self.container_port}/tcp"
            )
            if not port_info:
                await asyncio.sleep(0.5)
                continue

            host_port = port_info[0]["HostPort"]
            base_url = f"http://127.0.0.1:{host_port}"

            # Health check OpenCode server
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        return SandboxLease(
                            container_id=container.id,
                            base_url=base_url,
                            volume_name=volume_name,
                        )
            except Exception:
                pass

            await asyncio.sleep(0.5)

    def get_pool_stats(self) -> dict[str, int]:
        """Get current pool statistics.
        
        Returns:
            Dictionary with ready, in_use, and total counts
        """
        return {
            "ready": self._ready_queue.qsize(),
            "in_use": len(self._in_use),
            "total": self._ready_queue.qsize() + len(self._in_use),
            "capacity": self.max_total_size,
        }

    async def shutdown(self) -> None:
        """Shutdown the pool by destroying all containers and volumes."""
        logger.info("Shutting down sandbox pool...")
        
        # Drain ready queue
        containers_to_cleanup = []
        while not self._ready_queue.empty():
            try:
                lease = self._ready_queue.get_nowait()
                containers_to_cleanup.append(lease["container_id"])
            except asyncio.QueueEmpty:
                break

        # Add in-use containers
        containers_to_cleanup.extend(self._in_use)

        # Cleanup all containers
        for container_id in containers_to_cleanup:
            try:
                await self.release(container_id, healthy=False)
            except Exception as e:
                logger.error(f"Error cleaning up container {container_id[:12]}: {e}")

        logger.info("Sandbox pool shutdown complete")

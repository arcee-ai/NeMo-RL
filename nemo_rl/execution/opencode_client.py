"""OpenCode HTTP client for controlling sandboxes programmatically."""

import asyncio
import json
from typing import Any, Optional


import httpx


class OpenCodeClient:
    """Async HTTP client for OpenCode server API.

    OpenCode exposes a REST interface documented at ``/doc`` when the
    headless server is running (``opencode serve``). The important endpoints used
    here are:

    * ``POST /session`` – create shell session
    * ``DELETE /session/{id}`` – terminate a session
    * ``POST /session/{id}/shell`` – run shell commands
    * ``POST /file`` and ``GET /file`` – manage workspace files

    Args:
        base_url: Base URL of the OpenCode server (e.g., "http://127.0.0.1:4096")
        timeout: Default timeout for requests in seconds
        max_retries: Maximum number of retries for transient failures
        auth_token: Optional bearer token forwarded as ``Authorization`` header
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        auth_token: Optional[str] = None,
        default_agent: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "User-Agent": "nemo-rl-opencode-client/1.0",
        }
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"
        self._default_agent = default_agent

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
            headers=self._headers,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        retries: int = 0,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic for transient failures."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if retries < self.max_retries:
                await asyncio.sleep(min(2 ** retries, 10))
                return await self._request(method, endpoint, retries + 1, **kwargs)
            raise RuntimeError(f"Request failed after {self.max_retries} retries: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and retries < self.max_retries:
                await asyncio.sleep(min(2 ** retries, 10))
                return await self._request(method, endpoint, retries + 1, **kwargs)
            raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}") from e

    async def health_check(self) -> bool:
        """Check if the OpenCode server is healthy and responding."""
        try:
            response = await self._request("GET", "/doc")
            return response.status_code == 200
        except Exception:
            return False

    async def create_session(
        self,
        command: str = "bash",
        cwd: str = "/workspace",
        env: Optional[dict[str, str]] = None,
        name: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> str:
        """Create a new shell session.
        
        Args:
            command: Shell command to use (bash, zsh, sh, etc.)
            cwd: Working directory for the session
            env: Environment variables to set
            name: Optional session name
            
        Returns:
            Session ID
        """
        payload = {
            "command": command,
            "cwd": cwd,
            "env": env or {},
        }
        if name:
            payload["name"] = name
        agent_value = agent or self._default_agent
        if agent_value:
            payload["agent"] = agent_value

        response = await self._request("POST", "/session", json=payload)
        data = response.json()
        if isinstance(data, dict):
            if "id" in data:
                return data["id"]
            if "session" in data and isinstance(data["session"], dict):
                return data["session"].get("id")
        raise RuntimeError("OpenCode create_session response missing session id")

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        response = await self._request("GET", "/session")
        data = response.json()
        if isinstance(data, dict):
            if "sessions" in data and isinstance(data["sessions"], list):
                return data["sessions"]
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
        return []

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        await self._request("DELETE", f"/session/{session_id}")

    async def execute_command(
        self,
        session_id: str,
        command: str,
        timeout: Optional[float] = None,
        agent: Optional[str] = None,
    ) -> tuple[str, str, int]:
        """Execute a command in a session and wait for completion.
        
        Args:
            session_id: Session ID
            command: Command to execute
            timeout: Optional command timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        payload = {
            "command": command,
        }
        if timeout:
            payload["timeout"] = timeout
        agent_value = agent or self._default_agent
        if agent_value:
            payload["agent"] = agent_value

        response = await self._request("POST", f"/session/{session_id}/shell", json=payload)
        data = response.json()
        
        return (
            data.get("stdout", ""),
            data.get("stderr", ""),
            data.get("exit_code", 0),
        )

    async def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w",
        agent: Optional[str] = None,
    ) -> None:
        """Write content to a file.
        
        Args:
            path: File path (absolute or relative to session cwd)
            content: File content
            mode: Write mode ('w' for write, 'a' for append)
        """
        payload = {
            "path": path,
            "content": content,
            "mode": mode,
        }
        agent_value = agent or self._default_agent
        if agent_value:
            payload["agent"] = agent_value
        await self._request("POST", "/file", json=payload)

    async def read_file(self, path: str, agent: Optional[str] = None) -> str:
        """Read content from a file.
        
        Args:
            path: File path (absolute or relative to session cwd)
            
        Returns:
            File content
        """
        params = {"path": path}
        agent_value = agent or self._default_agent
        if agent_value:
            params["agent"] = agent_value

        response = await self._request("GET", "/file", params=params)
        data = response.json()
        if isinstance(data, dict):
            return data.get("content", "")
        return ""

    async def file_exists(self, path: str, agent: Optional[str] = None) -> bool:
        """Check if a file exists.
        
        Args:
            path: File path
            
        Returns:
            True if file exists, False otherwise
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/file"
        params = {"path": path}
        agent_value = agent or self._default_agent
        if agent_value:
            params["agent"] = agent_value

        response = await self._client.get(url, params=params)
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False

    async def authenticate(
        self,
        provider: str,
        credentials: dict[str, str],
    ) -> None:
        """Authenticate with external services (e.g., git, cloud providers).
        
        Args:
            provider: Provider name (e.g., "github", "gitlab", "aws")
            credentials: Provider-specific credentials
        """
        payload = {
            "provider": provider,
            "credentials": credentials,
        }
        await self._request("POST", "/auth", json=payload)

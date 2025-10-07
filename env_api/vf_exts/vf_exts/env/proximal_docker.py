# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal Docker environment for isolated code execution with OpenCode sandboxes."""

import asyncio
from contextlib import suppress
from datetime import datetime
import logging
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Callable, Optional

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

from nemo_rl.execution.events import RolloutCompleted
from nemo_rl.execution.opencode_client import OpenCodeClient
from nemo_rl.execution.reward import RewardAggregator
from nemo_rl.execution.sandbox_backends import SandboxSession, build_backend_sequence
from nemo_rl.execution.tracing import RolloutTracer


logger = logging.getLogger(__name__)


class ProximalDockerEnv(vf.MultiTurnEnv):
    """Verifiers environment that executes code in isolated Docker sandboxes.
    
    This environment integrates with NeMo-RL's GRPO training by:
    1. Using a Ray-managed pool of Docker containers running OpenCode server
    2. Acquiring exclusive sandboxes for each rollout sample
    3. Executing task setup, prompting, and testing inside containers
    4. Calling vLLM (via AsyncOpenAI) for policy generation
    5. Parsing test results (JUnit XML) to compute rewards
    6. Releasing sandboxes back to the pool for recycling
    
    The environment respects the generation_semaphore passed through vf_rollouts.py
    and supports concurrent execution across GRPO groups.
    
    Configuration:
        image: Docker image with OpenCode server (e.g., "opencode-worker:latest")
        warm_pool_size: Number of containers to keep ready (default: 16)
        max_total_size: Maximum total containers (default: 48)
        setup_command: Shell command to prepare the environment (e.g., "taskrunner setup")
        prompt_command: Shell command to get the task prompt (e.g., "taskrunner prompt")
        test_command: Shell command to run tests (e.g., "taskrunner test --xml")
        junit_path: Path to JUnit XML output (default: "/workspace/.trace/junit.xml")
        working_dir: Working directory for commands (default: "/workspace")
        shell: Shell to use for sessions (default: "bash")
        command_timeout: Timeout for individual commands in seconds (default: 300)
        
    Example config:
        environment_name: "vf_proximal_docker"
        environment_config:
            image: "ghcr.io/your-org/opencode-worker:latest"
            warm_pool_size: 16
            max_total_size: 48
            setup_command: "git clone https://github.com/your-org/tasks && cd tasks && pip install -e ."
            prompt_command: "python -m tasks.get_prompt"
            test_command: "pytest tests/ --junitxml=/workspace/.trace/junit.xml"
            junit_path: "/workspace/.trace/junit.xml"
    """

    def __init__(
        self,
        image: str,
        dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        warm_pool_size: int = 16,
        max_total_size: int = 48,
        setup_command: str = "taskrunner setup",
        prompt_command: str = "taskrunner prompt",
        test_command: str = "taskrunner test --xml",
        junit_path: str = "/workspace/.trace/junit.xml",
        working_dir: str = "/workspace",
        shell: str = "bash",
        command_timeout: float = 300.0,
        sandbox_user: str = "agent",
        max_turns: int = 1,
        trace_output_dir: Optional[str] = None,
        event_callback: Optional[Callable[[RolloutCompleted], Any]] = None,
        reward_weights: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        backend_priority = kwargs.pop("backend_priority", None)
        beam_config = kwargs.pop("beam_config", None)
        e2b_config = kwargs.pop("e2b_config", None)
        docker_config_overrides = kwargs.pop("docker_config", None)

        self.image = image
        self.warm_pool_size = warm_pool_size
        self.max_total_size = max_total_size
        self.setup_command = setup_command
        self.prompt_command = prompt_command
        self.test_command = test_command
        self.junit_path = junit_path
        self.working_dir = working_dir
        self.shell = shell
        self.command_timeout = command_timeout
        self.sandbox_user = sandbox_user

        if trace_output_dir is None:
            trace_dir = Path.cwd() / "rollout_traces"
        else:
            trace_dir = Path(trace_output_dir).expanduser()
        self.trace_output_dir = trace_dir.resolve()
        self.trace_output_dir.mkdir(parents=True, exist_ok=True)

        reward_weights = reward_weights or {}
        self._reward_aggregator = RewardAggregator(
            build_weight=reward_weights.get("build_ok", 0.2),
            pass_rate_weight=reward_weights.get("tests_passed_frac", 0.8),
        )
        self._event_callback = event_callback

        docker_overrides = docker_config_overrides or {}
        self._docker_backend_config = {
            "image": docker_overrides.get("image", image),
            "warm_pool_size": docker_overrides.get("warm_pool_size", warm_pool_size),
            "max_total_size": docker_overrides.get("max_total_size", max_total_size),
            "readiness_timeout": docker_overrides.get("readiness_timeout", 180.0),
            "container_port": docker_overrides.get("container_port", 4096),
        }

        priority = backend_priority or ["beam", "e2b", "docker"]
        try:
            self._sandbox_backends = build_backend_sequence(
                priority,
                docker_config=self._docker_backend_config,
                beam_config=beam_config,
                e2b_config=e2b_config,
            )
        except ValueError:
            # Fall back to Docker only if no valid configuration is found
            self._sandbox_backends = build_backend_sequence(
                ["docker"],
                docker_config=self._docker_backend_config,
                beam_config=None,
                e2b_config=None,
            )

        self._active_backend: Optional[str] = None
        self._vllm_client: Optional[AsyncOpenAI] = None

        super().__init__(
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        backend_names = ",".join(name for name, _ in self._sandbox_backends)
        logger.info(
            "Initialized ProximalDockerEnv with backends %s (image=%s, warm_pool=%s, max=%s)",
            backend_names,
            image,
            warm_pool_size,
            max_total_size,
        )

    def _get_vllm_client(self) -> AsyncOpenAI:
        """Get or create the vLLM AsyncOpenAI client."""
        if self._vllm_client is None:
            self._vllm_client = AsyncOpenAI(
                api_key="n/a",
                base_url="http://127.0.0.1:8000/v1",
            )
        return self._vllm_client

    async def _acquire_sandbox_session(self) -> SandboxSession:
        """Acquire a sandbox session from the configured backends."""
        last_exc: Optional[BaseException] = None
        for backend_name, backend in self._sandbox_backends:
            try:
                session = await backend.acquire()
                self._active_backend = backend_name
                return session
            except BaseException as exc:  # pragma: no cover - best effort logging
                last_exc = exc
                logger.warning(
                    "Failed to acquire sandbox via backend '%s': %s",
                    backend_name,
                    exc,
                )

        raise RuntimeError("Failed to acquire sandbox from all configured backends") from last_exc

    async def a_generate(
        self,
        inputs: vf.GenerateInputs | Dataset | dict,
        sampling_args: vf.SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        client: Optional[AsyncOpenAI] = None,
        model: str = "policy",
        **kwargs,
    ) -> tuple[vf.GenerateOutputs, vf.ProcessedOutputs]:
        """Generate rollouts by executing tasks in isolated Docker sandboxes.
        
        This method:
        1. Acquires a sandbox from the pool for each input sample
        2. Runs setup and gets the task prompt
        3. Calls vLLM to generate a solution
        4. Applies the solution and runs tests
        5. Parses test results to compute rewards
        6. Releases sandboxes back to the pool
        
        Concurrency is controlled by max_concurrent (from generation_semaphore).
        """
        vllm_client = client or self._get_vllm_client()

        # Convert inputs to standard format
        if isinstance(inputs, dict):
            prompts = inputs.get("prompt", [])
            answers = inputs.get("answer", [])
            infos = inputs.get("info", [])
            tasks = inputs.get("task", [])
        elif isinstance(inputs, Dataset):
            prompts = list(inputs["prompt"]) if "prompt" in inputs.column_names else []
            answers = list(inputs["answer"]) if "answer" in inputs.column_names else []
            infos = list(inputs["info"]) if "info" in inputs.column_names else []
            tasks = list(inputs["task"]) if "task" in inputs.column_names else []
        else:
            # Handle GenerateInputs or pre-built prompt lists
            prompts = inputs if isinstance(inputs, list) else inputs.prompt
            answers = [""] * len(prompts)
            infos = [{}] * len(prompts)
            tasks = ["default"] * len(prompts)

        prompts = list(prompts)
        total = len(prompts)

        def _ensure_list(items: Any, default_factory: Callable[[], Any]) -> list[Any]:
            seq = list(items) if items is not None else []
            if len(seq) == total:
                return seq
            if len(seq) == 0:
                return [default_factory() for _ in range(total)]
            if len(seq) < total:
                filler = [default_factory() for _ in range(total - len(seq))]
                return seq + filler
            return seq[:total]

        answers = _ensure_list(answers, lambda: "")
        infos = [dict(item) if isinstance(item, dict) else {} for item in _ensure_list(infos, dict)]
        tasks = [str(item) if isinstance(item, str) and item else "default" for item in _ensure_list(tasks, lambda: "default")]

        if total == 0:
            return vf.GenerateOutputs(
                prompt=[],
                completion=[],
                answer=[],
                state=[],
                info=[],
                task=[],
                reward=[],
                metrics={},
            ), vf.ProcessedOutputs(
                prompt_ids=[],
                prompt_mask=[],
                completion_ids=[],
                completion_mask=[],
                completion_logprobs=[],
                rewards=[],
            )

        states = [self.reset_for_rollout(prompts[i], answers[i], infos[i], tasks[i]) for i in range(total)]

        # Use semaphore to respect max_concurrent
        sem = asyncio.Semaphore(max_concurrent if max_concurrent > 0 else len(prompts))

        async def run_one(i: int) -> tuple[vf.Messages, list[dict], float, dict, vf.State]:
            """Execute one rollout in an isolated sandbox."""
            async with sem:
                return await self._run_single_rollout(
                    prompt=prompts[i],
                    answer=answers[i],
                    info=infos[i],
                    task=tasks[i],
                    state=states[i],
                    client=vllm_client,
                    model=model,
                    sampling_args=sampling_args or {},
                )

        # Execute all rollouts concurrently
        results = await asyncio.gather(
            *(run_one(i) for i in range(len(prompts))),
            return_exceptions=False,
        )

        used_prompts, completions, rewards, metric_list, final_states = zip(*results)

        metrics_map = self._collect_metrics(list(metric_list))

        generate_outputs = vf.GenerateOutputs(
            prompt=list(used_prompts),
            completion=list(completions),
            answer=list(answers),
            state=list(final_states),
            info=list(infos),
            task=list(tasks),
            reward=list(rewards),
            metrics=metrics_map,
        )

        processed_outputs = vf.ProcessedOutputs(
            prompt_ids=[[] for _ in completions],
            prompt_mask=[[] for _ in completions],
            completion_ids=[[] for _ in completions],
            completion_mask=[[] for _ in completions],
            completion_logprobs=[[] for _ in completions],
            rewards=list(rewards),
        )

        return generate_outputs, processed_outputs

    async def _run_single_rollout(
        self,
        prompt: vf.Messages,
        answer: str,
        info: vf.Info,
        task: str,
        state: vf.State,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict,
    ) -> tuple[vf.Messages, list[dict], float, dict, vf.State]:
        """Execute a single rollout in an isolated sandbox."""
        session: Optional[SandboxSession] = None
        session_id: Optional[str] = None
        release_healthy = False
        stage = "acquire"
        prompt_messages = self._prepare_prompt(prompt)
        completion_messages = self._empty_completion()
        container_prompt = ""
        backend_name = None
        tracer: Optional[RolloutTracer] = None
        episode_id = str(info.get("episode_id") or uuid.uuid4())
        test_exit_code: Optional[int] = None
        rollout_state: vf.State = dict(state or {})
        rollout_state["episode_id"] = episode_id

        try:
            session = await self._acquire_sandbox_session()
            backend_name = self._active_backend
            stage = "connect"
            rollout_state["sandbox_backend"] = backend_name

            agent_user = self._resolve_agent_user(info)

            async with OpenCodeClient(session.base_url, default_agent=agent_user) as oc_client:
                session_id = await oc_client.create_session(
                    command=self.shell,
                    cwd=self.working_dir,
                    name=f"rollout-{task}",
                    agent=agent_user,
                )
                stage = "setup"
                rollout_state["stage"] = stage

                tracer = RolloutTracer(
                    episode_id=episode_id,
                    sandbox_backend=backend_name or "unknown",
                    policy_id=model,
                    artifact_root=self.trace_output_dir,
                    event_callback=self._event_callback,
                )
                tracer.set_prompt(prompt_messages)

                try:
                    setup_cmd = self._build_command(self.setup_command, info)
                    if setup_cmd:
                        await self._execute_command(
                            oc_client,
                            session_id,
                            setup_cmd,
                            stage="setup",
                            tracer=tracer,
                        )

                    stage = "prompt"
                    rollout_state["stage"] = stage
                    prompt_cmd = self._build_command(self.prompt_command, info)
                    if prompt_cmd:
                        prompt_stdout, _, _ = await self._execute_command(
                            oc_client,
                            session_id,
                            prompt_cmd,
                            stage="prompt",
                            tracer=tracer,
                        )
                        container_prompt = prompt_stdout.strip()
                        if container_prompt:
                            prompt_messages = self._merge_prompt_with_container(
                                prompt_messages,
                                container_prompt,
                                info,
                            )
                            tracer.set_prompt(prompt_messages)
                            rollout_state["container_prompt"] = container_prompt
                            rollout_state["prompt"] = prompt_messages

                    stage = "generation"
                    rollout_state["stage"] = stage
                    completion_messages = await self._generate_with_vllm(
                        client=client,
                        model=model,
                        prompt=prompt_messages,
                        sampling_args=sampling_args,
                    )
                    tracer.set_completion(completion_messages)
                    rollout_state["completion"] = completion_messages

                    solution_text = self._extract_solution_text(completion_messages)
                    stage = "apply_solution"
                    rollout_state["stage"] = stage
                    await self._apply_solution(
                        oc_client,
                        session_id,
                        solution_text,
                        info,
                        tracer=tracer,
                    )

                    stage = "test"
                    rollout_state["stage"] = stage
                    test_cmd = self._build_command(self.test_command, info)
                    if test_cmd:
                        _, _, test_exit_code = await self._execute_command(
                            oc_client,
                            session_id,
                            test_cmd,
                            stage="test",
                            tracer=tracer,
                            expect_success=False,
                        )
                    else:
                        test_exit_code = 0

                    stage = "parse_results"
                    rollout_state["stage"] = stage
                    pass_rate, test_metrics = await self._parse_junit_results(oc_client)
                    build_success = (test_exit_code == 0) if test_exit_code is not None else False
                    reward_breakdown = self._reward_aggregator.compute(
                        pass_rate=pass_rate,
                        build_succeeded=build_success,
                    )
                    reward_payload = reward_breakdown.to_payload()
                    tracer.set_reward(reward_payload)
                    rollout_state["reward"] = reward_payload
                    rollout_state["pass_rate"] = pass_rate
                    rollout_state["build_exit_code"] = test_exit_code
                    rollout_state["stage"] = "completed"
                    test_metrics.update(
                        {
                            "status_success": 1,
                            "prompt_chars": len(container_prompt),
                            "completion_chars": len(solution_text),
                            "sandbox_backend": backend_name or "unknown",
                            "build_exit_code": test_exit_code if test_exit_code is not None else -1,
                            "resolved_fraction": reward_breakdown.resolved_fraction,
                            "reward_build_component": reward_breakdown.build_ok,
                            "reward_pass_component": reward_breakdown.tests_passed_frac,
                            "episode_id": episode_id,
                        }
                    )
                    test_metrics.update({f"reward_shaping_{k}": v for k, v in reward_payload["shaping"].items()})
                    tracer.set_metrics(test_metrics)
                    tracer.finished()
                    artifacts: dict[str, str] = {}
                    try:
                        artifacts = tracer.flush()
                    except Exception as trace_exc:  # pragma: no cover - best effort
                        logger.exception("Failed to flush rollout trace for %s", tracer.episode_id)
                    try:
                        tracer.emit(artifacts)
                    except Exception as emit_exc:  # pragma: no cover - best effort
                        logger.exception("Failed to emit rollout event for %s", tracer.episode_id)
                    release_healthy = True
                    test_metrics["artifact_dir"] = artifacts.get("metadata", "") if artifacts else ""
                    rollout_state["artifact_dir"] = test_metrics["artifact_dir"]
                    rollout_state["turn_count"] = rollout_state.get("turn_count", 0) + 1
                    rollout_state["last_success"] = True
                    return (
                        prompt_messages,
                        completion_messages,
                        reward_breakdown.resolved_fraction,
                        test_metrics,
                        rollout_state,
                    )

                except Exception as exc:
                    logger.exception("Rollout failed during %s stage", stage)
                    failure_metrics = self._build_failure_metrics(stage, fatal=False)
                    failure_metrics["sandbox_backend"] = backend_name or "unknown"
                    failure_metrics["episode_id"] = episode_id
                    rollout_state["stage"] = stage
                    rollout_state["error"] = str(exc)
                    if tracer:
                        tracer.set_error(str(exc))
                        tracer.finished()
                        artifacts: dict[str, str] = {}
                        try:
                            artifacts = tracer.flush()
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("Failed to flush rollout trace for %s", tracer.episode_id)
                        try:
                            tracer.emit(artifacts)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("Failed to emit rollout event for %s", tracer.episode_id)
                        failure_metrics["artifact_dir"] = artifacts.get("metadata", "") if artifacts else ""
                    rollout_state["artifact_dir"] = failure_metrics.get("artifact_dir", "")
                    rollout_state["turn_count"] = rollout_state.get("turn_count", 0) + 1
                    rollout_state["last_success"] = False
                    return prompt_messages, completion_messages, 0.0, failure_metrics, rollout_state

                finally:
                    if session_id:
                        with suppress(Exception):
                            await oc_client.delete_session(session_id)

        except Exception as exc:
            logger.exception("Rollout failed: %s", exc)
            failure_metrics = self._build_failure_metrics(stage, fatal=True)
            failure_metrics["sandbox_backend"] = backend_name or "unknown"
            failure_metrics["episode_id"] = episode_id
            rollout_state["stage"] = stage
            rollout_state["error"] = str(exc)
            if tracer:
                tracer.set_error(str(exc))
                tracer.finished()
                artifacts: dict[str, str] = {}
                try:
                    artifacts = tracer.flush()
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to flush rollout trace for %s", tracer.episode_id)
                try:
                    tracer.emit(artifacts)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to emit rollout event for %s", tracer.episode_id)
                failure_metrics["artifact_dir"] = artifacts.get("metadata", "") if artifacts else ""
            rollout_state["artifact_dir"] = failure_metrics.get("artifact_dir", "")
            rollout_state["turn_count"] = rollout_state.get("turn_count", 0) + 1
            rollout_state["last_success"] = False
            return prompt_messages, completion_messages, 0.0, failure_metrics, rollout_state
        finally:
            if session is not None:
                with suppress(Exception):
                    await session.release(release_healthy)

    def _prepare_prompt(self, prompt: vf.Messages | str) -> vf.Messages:
        """Return a copy of the prompt in chat message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        copied = []
        for message in prompt:
            msg_copy = dict(message)
            msg_copy.setdefault("role", "user")
            msg_copy.setdefault("content", "")
            copied.append(msg_copy)
        return copied

    def _empty_completion(self) -> list[dict[str, Any]]:
        """Return a placeholder completion structure."""
        return [{"role": "assistant", "content": ""}]

    def _merge_prompt_with_container(
        self,
        prompt: vf.Messages,
        container_prompt: str,
        info: vf.Info,
    ) -> vf.Messages:
        """Inject the prompt returned by the container into the message list."""
        merged = [dict(message) for message in prompt]
        prompt_prefix = info.get("prompt_prefix")
        prompt_suffix = info.get("prompt_suffix")

        text = container_prompt.strip()
        if prompt_prefix:
            text = f"{prompt_prefix}\n\n{text}"
        if prompt_suffix:
            text = f"{text}\n\n{prompt_suffix}"

        if merged and merged[-1].get("role") == "user":
            merged[-1]["content"] = text
        else:
            merged.append({"role": "user", "content": text})
        return merged

    def _build_command(self, base_command: str, info: vf.Info) -> str:
        """Augment a base command with problem/variant flags or template values."""
        problem_id = info.get("problem_id")
        variant_id = info.get("variant_id")
        extra_args = info.get("taskrunner_args", [])

        command = base_command or ""
        if command:
            format_values = {
                "problem_id": problem_id or "",
                "variant_id": variant_id or "",
                "task": info.get("task") or "",
            }

            try:
                command = command.format(**format_values)
            except KeyError:
                pass

        tokens: list[str] = [command] if command else []

        if problem_id and (not command or "--problem" not in command):
            tokens.append(f"--problem {problem_id}")
        if variant_id and (not command or "--variant" not in command):
            tokens.append(f"--variant {variant_id}")

        if isinstance(extra_args, str) and extra_args:
            tokens.append(extra_args)
        elif isinstance(extra_args, (list, tuple)):
            tokens.extend(str(arg) for arg in extra_args if arg)

        return " ".join(token for token in tokens if token)

    def _build_failure_metrics(self, stage: str, fatal: bool) -> dict[str, Any]:
        """Return a metrics dictionary describing a rollout failure."""
        return {
            "status_failure": 1,
            "error_stage": stage,
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_errors": 0,
            "tests_skipped": 0,
            "pass_rate": 0.0,
            "prompt_chars": 0,
            "completion_chars": 0,
            "rollout_error": 1,
            "rollout_error_fatal": 1 if fatal else 0,
        }

    def _collect_metrics(self, metrics_list: list[dict[str, Any]]) -> dict[str, list[float]]:
        if not metrics_list:
            return {}

        numeric_keys: set[str] = set()
        for metric in metrics_list:
            for key, value in metric.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)

        if not numeric_keys:
            return {}

        collected: dict[str, list[float]] = {key: [] for key in numeric_keys}

        for metric in metrics_list:
            for key in numeric_keys:
                value = metric.get(key)
                if isinstance(value, (int, float)):
                    collected[key].append(float(value))
                else:
                    collected[key].append(0.0)

        return collected

    def _resolve_agent_user(self, info: vf.Info) -> str:
        if isinstance(info, dict):
            for key in ("sandbox_user", "agent_user", "user"):
                candidate = info.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return self.sandbox_user

    def reset_for_rollout(
        self,
        prompt: vf.Messages,
        answer: str,
        info: vf.Info | None = None,
        task: str | None = None,
    ) -> vf.State:
        info_dict: dict[str, Any] = dict(info or {})
        resolved_task = task or info_dict.get("task") or "default"
        state: vf.State = {
            "prompt": prompt,
            "answer": answer,
            "info": info_dict,
            "task": resolved_task,
            "responses": [],
            "turn_count": 0,
            "history": [],
            "stage": "initialized",
        }
        return self.setup_state(state)

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state.setdefault("responses", [])
        state.setdefault("turn_count", 0)
        state.setdefault("history", [])
        state.setdefault("stage", "initialized")
        if "task" not in state:
            state["task"] = "default"
        return state

    async def _execute_command(
        self,
        client: OpenCodeClient,
        session_id: str,
        command: str,
        stage: str,
        expect_success: bool = True,
        tracer: Optional[RolloutTracer] = None,
    ) -> tuple[str, str, int]:
        """Execute a command and log the result."""
        logger.debug(f"Executing {stage}: {command}")
        started_at = datetime.utcnow()
        stdout, stderr, exit_code = await client.execute_command(
            session_id=session_id,
            command=command,
            timeout=self.command_timeout,
        )
        completed_at = datetime.utcnow()

        if tracer:
            tracer.record_command(
                stage=stage,
                command=command,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )

        if exit_code != 0:
            logger.warning(
                f"{stage} command failed (exit {exit_code}):\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )
            if expect_success:
                raise RuntimeError(f"Command '{command}' failed during {stage} (exit {exit_code})")
        else:
            logger.debug(f"{stage} command succeeded")

        return stdout, stderr, exit_code

    async def _generate_with_vllm(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: vf.Messages,
        sampling_args: dict,
    ) -> list[dict]:
        """Generate completion using vLLM."""
        # Convert sampling_args to OpenAI format
        openai_args = {
            "temperature": sampling_args.get("temperature", 0.7),
            "top_p": sampling_args.get("top_p", 0.9),
            "max_tokens": sampling_args.get("max_tokens", 512),
            "logprobs": sampling_args.get("logprobs", True),
        }

        # Remove None values
        openai_args = {k: v for k, v in openai_args.items() if v is not None}

        # Call vLLM
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            **openai_args,
        )

        # Extract completion
        completion_message = response.choices[0].message
        return [
            {
                "role": "assistant",
                "content": completion_message.content or "",
                "tool_calls": getattr(completion_message, "tool_calls", None) or [],
            }
        ]

    def _extract_solution_text(self, completion_messages: list[dict]) -> str:
        """Extract solution text from completion messages."""
        if not completion_messages:
            return ""
        return completion_messages[0].get("content", "")

    async def _apply_solution(
        self,
        client: OpenCodeClient,
        session_id: str,
        solution_text: str,
        info: vf.Info,
        tracer: Optional[RolloutTracer] = None,
    ) -> None:
        """Apply the generated solution to the workspace.
        
        This method writes the solution to a file and applies it.
        You may need to customize this based on your task format.
        """
        # Write solution to file
        solution_path = f"{self.working_dir}/solution.txt"
        await client.write_file(solution_path, solution_text)
        if tracer:
            tracer.record_command(
                stage="apply_solution.write",
                command=f"write_file {solution_path}",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                exit_code=0,
                stdout="",
                stderr="",
            )

        # If there's an apply command in info, use it
        apply_command = info.get("apply_command")
        if apply_command:
            await self._execute_command(
                client,
                session_id,
                self._build_command(apply_command, info),
                "apply_solution.exec",
                tracer=tracer,
            )

    async def _parse_junit_results(
        self,
        client: OpenCodeClient,
    ) -> tuple[float, dict]:
        """Parse JUnit XML results to compute reward.
        
        Returns:
            Tuple of (reward, metrics_dict)
        """
        try:
            # Read JUnit XML file
            xml_content = await client.read_file(self.junit_path)
            if not xml_content.strip():
                raise FileNotFoundError(self.junit_path)
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract test statistics
            if root.tag == "testsuites":
                testsuite = root.find("testsuite")
                if testsuite is None:
                    testsuite = root
            else:
                testsuite = root

            tests = int(testsuite.get("tests", 0))
            failures = int(testsuite.get("failures", 0))
            errors = int(testsuite.get("errors", 0))
            skipped = int(testsuite.get("skipped", 0))

            passed = tests - failures - errors - skipped

            # Compute reward as pass rate
            if tests > 0:
                reward = passed / tests
            else:
                reward = 0.0

            metrics = {
                "tests_total": tests,
                "tests_passed": passed,
                "tests_failed": failures,
                "tests_errors": errors,
                "tests_skipped": skipped,
                "pass_rate": reward,
            }

            logger.debug(f"Test results: {passed}/{tests} passed (reward={reward:.2f})")

            return reward, metrics

        except FileNotFoundError:
            logger.warning(f"JUnit XML not found at {self.junit_path}, reward=0")
            return 0.0, {
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_errors": 0,
                "tests_skipped": 0,
                "pass_rate": 0.0,
                "missing_junit": 1,
            }
        except Exception as e:
            logger.error(f"Failed to parse JUnit XML: {e}")
            return 0.0, {
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_errors": 0,
                "tests_skipped": 0,
                "pass_rate": 0.0,
                "parse_error": 1,
            }


    def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs
    ) -> tuple[vf.Messages, vf.State]:
        """Return environment response (not used in ProximalDockerEnv).
        
        This environment uses a_generate() for the full rollout flow instead
        of the traditional MultiTurnEnv message-by-message interaction.
        """
        # Return empty response - actual interaction happens via a_generate()
        return [], state

    def is_completed(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs
    ) -> bool:
        """Check if the rollout is completed.
        
        For ProximalDockerEnv, rollouts are single-turn (max_turns=1 by default),
        so they complete after one generation.
        """
        # Check if we've done at least one turn
        turn_count = state.get("turn_count", 0)
        return turn_count >= self.max_turns


def load_environment(**kwargs) -> ProximalDockerEnv:
    """Factory function for loading the environment via vf.load_environment()."""
    return ProximalDockerEnv(**kwargs)

"""Utilities for recording sandbox rollout traces and artifacts."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from nemo_rl.execution.events import CommandLog, RolloutCompleted


@dataclass(slots=True)
class RolloutTrace:
    """Container for command logs and associated metadata."""

    episode_id: str
    sandbox_backend: str
    artifact_dir: Optional[Path]
    prompt: Optional[Any] = None
    completion: Optional[Any] = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    command_logs: List[CommandLog] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    reward_package: Dict[str, Any] = field(default_factory=dict)

    def add_command(
        self,
        stage: str,
        command: str,
        started_at: datetime,
        completed_at: datetime,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command_logs.append(
            CommandLog(
                stage=stage,
                command=command,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
        )

    def as_rollout_completed(self, policy_id: str, artifact_uris: Dict[str, str]) -> RolloutCompleted:
        return RolloutCompleted(
            episode_id=self.episode_id,
            policy_id=policy_id,
            reward=self.reward_package,
            metrics=self.metrics,
            artifact_uris=artifact_uris,
            sandbox_backend=self.sandbox_backend,
            commands=list(self.command_logs),
            prompt=self.prompt,
            completion=self.completion,
            error=self.error,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


class RolloutTracer:
    """Records rollout activity and optionally writes artifacts to disk."""

    def __init__(
        self,
        episode_id: str,
        sandbox_backend: str,
        policy_id: str,
        artifact_root: Optional[Path] = None,
        event_callback: Optional[Callable[[RolloutCompleted], Any]] = None,
    ) -> None:
        self._trace = RolloutTrace(
            episode_id=episode_id,
            sandbox_backend=sandbox_backend,
            artifact_dir=artifact_root,
        )
        self._policy_id = policy_id
        self._artifact_root = artifact_root
        self._event_callback = event_callback

    @property
    def episode_id(self) -> str:
        return self._trace.episode_id

    def set_prompt(self, prompt: Any) -> None:
        self._trace.prompt = prompt

    def set_completion(self, completion: Any) -> None:
        self._trace.completion = completion

    def set_metrics(self, metrics: Dict[str, float]) -> None:
        self._trace.metrics = metrics

    def set_reward(self, reward: Dict[str, Any]) -> None:
        self._trace.reward_package = reward

    def set_error(self, error: str) -> None:
        self._trace.error = error

    def finished(self) -> None:
        self._trace.completed_at = datetime.utcnow()

    def record_command(
        self,
        stage: str,
        command: str,
        started_at: datetime,
        completed_at: datetime,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self._trace.add_command(stage, command, started_at, completed_at, exit_code, stdout, stderr)

    def _artifact_directory(self) -> Optional[Path]:
        if self._artifact_root is None:
            return None
        rollout_dir = self._artifact_root / self.episode_id
        rollout_dir.mkdir(parents=True, exist_ok=True)
        return rollout_dir

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def flush(self) -> Dict[str, str]:
        artifact_dir = self._artifact_directory()
        artifact_uris: Dict[str, str] = {}

        if artifact_dir is None:
            return artifact_uris

        commands_path = artifact_dir / "commands.jsonl"
        with commands_path.open("w", encoding="utf-8") as fp:
            for record in self._trace.command_logs:
                fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        artifact_uris["commands"] = str(commands_path)

        metadata_path = artifact_dir / "metadata.json"
        self._write_json(
            metadata_path,
            {
                "episode_id": self._trace.episode_id,
                "sandbox_backend": self._trace.sandbox_backend,
                "started_at": self._trace.started_at.isoformat(),
                "completed_at": self._trace.completed_at.isoformat() if self._trace.completed_at else None,
                "metrics": self._trace.metrics,
                "reward": self._trace.reward_package,
                "error": self._trace.error,
            },
        )
        artifact_uris["metadata"] = str(metadata_path)

        if self._trace.prompt is not None:
            prompt_path = artifact_dir / "prompt.json"
            self._write_json(prompt_path, self._trace.prompt)
            artifact_uris["prompt"] = str(prompt_path)

        if self._trace.completion is not None:
            completion_path = artifact_dir / "completion.json"
            self._write_json(completion_path, self._trace.completion)
            artifact_uris["completion"] = str(completion_path)

        return artifact_uris

    def emit(self, artifact_uris: Dict[str, str]) -> None:
        event = self._trace.as_rollout_completed(self._policy_id, artifact_uris)
        if self._event_callback:
            result = self._event_callback(event)
            if inspect.isawaitable(result):  # pragma: no branch - best effort
                asyncio.create_task(result)


def merge_metric_lists(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for metric in metric_dicts:
        for key, value in metric.items():
            if not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}

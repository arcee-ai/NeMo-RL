"""Event and artifact models for sandbox rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RolloutRequested:
    """Envelope emitted by verifiers when a rollout should start."""

    episode_id: str
    benchmark: str
    instance_id: str
    repo_ref: Dict[str, Any]
    time_budget_s: float
    policy: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommandLog:
    """Trace entry for an individual command executed in a sandbox."""

    stage: str
    command: str
    started_at: datetime
    completed_at: datetime
    exit_code: int
    stdout: str
    stderr: str

    @property
    def duration_s(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        data["duration_s"] = self.duration_s
        return data


@dataclass(slots=True)
class RolloutCompleted:
    """Summary emitted when a rollout finishes (success or failure)."""

    episode_id: str
    policy_id: str
    reward: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_uris: Dict[str, str]
    sandbox_backend: str
    commands: List[CommandLog] = field(default_factory=list)
    prompt: Optional[Any] = None
    completion: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["commands"] = [entry.to_dict() for entry in self.commands]
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


@dataclass(slots=True)
class RewardComputed:
    """Reward package handed back to verifiers/trainers."""

    episode_id: str
    policy: str
    reward: Dict[str, Any]
    artifact_uris: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

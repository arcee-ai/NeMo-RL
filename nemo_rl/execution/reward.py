"""Reward aggregation utilities for sandbox rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(slots=True)
class RewardBreakdown:
    """Structured reward package with shaping components."""

    resolved_fraction: float
    build_ok: float
    tests_passed_frac: float
    extra: Dict[str, float]

    def to_payload(self) -> Dict[str, Dict[str, float]]:
        shaping = {
            "build_ok": self.build_ok,
            "tests_passed_frac": self.tests_passed_frac,
        }
        shaping.update(self.extra)
        return {
            "unit": "resolved_fraction",
            "value": self.resolved_fraction,
            "shaping": shaping,
        }


class RewardAggregator:
    """Computes terminal and shaping rewards from test metrics."""

    def __init__(
        self,
        build_weight: float = 0.2,
        pass_rate_weight: float = 0.8,
    ) -> None:
        self.build_weight = build_weight
        self.pass_rate_weight = pass_rate_weight

    def compute(
        self,
        pass_rate: float,
        build_succeeded: bool,
        extras: Dict[str, float] | None = None,
    ) -> RewardBreakdown:
        extras = extras or {}
        build_component = self.build_weight if build_succeeded else 0.0
        pass_component = self.pass_rate_weight * max(pass_rate, 0.0)
        resolved_fraction = min(1.0, build_component + pass_component)
        return RewardBreakdown(
            resolved_fraction=resolved_fraction,
            build_ok=build_component,
            tests_passed_frac=pass_component,
            extra=extras,
        )

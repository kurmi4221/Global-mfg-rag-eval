"""Release policy: versioned in git, loaded from YAML, validated by pydantic."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TierPolicy(_Strict):
    # Failing any of these metrics on an S1 case is a critical failure.
    critical_metrics: list[str]
    s1_failures_allowed: int = 0
    # Must be present, finite, within floors, and must not regress vs baseline => else FAIL.
    gate_metrics: list[str]
    # Soft targets: regression => REVIEW (a named approver decides).
    guardrail_metrics: list[str] = Field(default_factory=list)
    # Absolute minimums that hold regardless of baseline (prevents slow decay).
    floors: dict[str, float] = Field(default_factory=dict)
    # Absolute maximums for operational SLOs; breach => REVIEW.
    ceilings: dict[str, float] = Field(default_factory=dict)
    approvals: list[str] = Field(default_factory=list)


class RegressionPolicy(_Strict):
    default_tolerance: float = 0.03
    # Per-metric absolute tolerances, calibrated as 2*sigma over N unchanged baseline runs.
    tolerances: dict[str, float] = Field(default_factory=dict)
    # Operational metrics (lower is better) use a relative tolerance.
    relative_tolerance_ops: float = 0.25
    # Latency also gets an absolute floor so micro-benchmarks do not flap on noise.
    ops_latency_abs_tolerance_ms: float = 50.0
    # Gate metrics are also compared with the median of the last N approved releases.
    history_window: int = 3


class Policy(_Strict):
    policy_version: str
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {"S1": 10.0, "S2": 3.0, "S3": 1.0}
    )
    regression: RegressionPolicy = Field(default_factory=RegressionPolicy)
    tiers: dict[str, TierPolicy]


def load_policy(path: str | Path) -> Policy:
    with Path(path).open(encoding="utf-8") as f:
        return Policy.model_validate(yaml.safe_load(f))

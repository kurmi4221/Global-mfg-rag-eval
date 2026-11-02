"""The release decision. FAIL-CLOSED by construction.

Order of evaluation (any FAIL wins, then any REVIEW, else PASS):

  1. S1 critical failure on a critical metric              -> FAIL
  2. pipeline/judge errors during the run                  -> FAIL
  3. a gate metric missing or NaN                          -> FAIL   (fail-closed)
  4. absolute floor violated                               -> FAIL
  5. gate metric regressed vs baseline beyond tolerance    -> FAIL
  6. gate metric below median of last N approved releases  -> FAIL   (slow-decay detector)
  7. guardrail metric regressed / ops SLO ceiling breached -> REVIEW
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, Field

from mfg_rag_eval.gating.policy import Policy

Snapshot = dict[str, Any]
Level = Literal["FAIL", "REVIEW", "INFO"]

EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


class Finding(BaseModel):
    level: Level
    code: str
    metric: str | None = None
    message: str


class Verdict(BaseModel):
    verdict: Literal["PASS", "REVIEW", "FAIL"]
    tier: str
    findings: list[Finding] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


def _finite(x: object) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _lower_is_better(metric_id: str) -> bool:
    return (
        metric_id.startswith("ops.latency")
        or metric_id.startswith("ops.tokens")
        or (metric_id == "run.errors")
    )


def _tolerance(policy: Policy, metric_id: str, baseline: float) -> float:
    if metric_id in policy.regression.tolerances:
        return policy.regression.tolerances[metric_id]
    if metric_id.startswith("ops."):
        rel = policy.regression.relative_tolerance_ops * abs(baseline)
        if metric_id.startswith("ops.latency"):
            return max(rel, policy.regression.ops_latency_abs_tolerance_ms)
        return rel
    return policy.regression.default_tolerance


def _worsening(metric_id: str, ref: float, cand: float) -> float:
    """Positive number = candidate is worse than the reference."""
    return (cand - ref) if _lower_is_better(metric_id) else (ref - cand)


def decide(
    policy: Policy,
    tier: str,
    candidate: Snapshot,
    baseline: Snapshot | None = None,
    history: list[Snapshot] | None = None,
) -> Verdict:
    tp = policy.tiers[tier]
    cm: dict[str, Any] = candidate.get("metrics", {})
    findings: list[Finding] = []
    rows: list[dict[str, Any]] = []

    def add(level: Level, code: str, message: str, metric: str | None = None) -> None:
        findings.append(Finding(level=level, code=code, metric=metric, message=message))

    # 1. critical S1 failures
    crit = candidate.get("s1_critical_failures", [])
    if len(crit) > tp.s1_failures_allowed:
        add(
            "FAIL",
            "S1_CRITICAL_FAILURE",
            f"{len(crit)} S1 critical failure(s) (allowed {tp.s1_failures_allowed}): {crit}",
        )

    # 2. run errors (also covers a missing counter => malformed snapshot)
    errors = cm.get("run.errors")
    if not _finite(errors):
        add(
            "FAIL",
            "MALFORMED_SNAPSHOT",
            "run.errors missing: cannot prove the run completed",
            "run.errors",
        )
    elif errors and errors > 0:
        add("FAIL", "RUN_ERRORS", f"{int(errors)} case(s) errored during the run", "run.errors")

    # 3. gate metrics must exist and be finite
    for mid in tp.gate_metrics:
        if not _finite(cm.get(mid)):
            add(
                "FAIL",
                "MISSING_GATE_METRIC",
                f"gate metric absent or NaN (value={cm.get(mid)!r}); failing closed",
                mid,
            )

    # 4. absolute floors
    for mid, floor in tp.floors.items():
        v = cm.get(mid)
        if not _finite(v):
            add("FAIL", "MISSING_FLOOR_METRIC", f"floor metric absent or NaN (value={v!r})", mid)
        elif v < floor:
            add("FAIL", "FLOOR_VIOLATION", f"{v:.3f} < floor {floor:.3f}", mid)

    # ops ceilings -> REVIEW
    for mid, ceiling in tp.ceilings.items():
        v = cm.get(mid)
        if _finite(v) and v > ceiling:
            add("REVIEW", "SLO_CEILING", f"{v:.3f} > ceiling {ceiling:.3f}", mid)

    # 5/7. regression vs baseline
    if baseline is None:
        add("INFO", "NO_BASELINE", "no baseline supplied: only absolute floors were enforced")
    else:
        bm: dict[str, Any] = baseline.get("metrics", {})
        watched: list[tuple[str, Level]] = [(m, "FAIL") for m in tp.gate_metrics]
        watched += [(m, "REVIEW") for m in tp.guardrail_metrics]
        for mid, level in watched:
            b, c = bm.get(mid), cm.get(mid)
            row: dict[str, Any] = {
                "metric": mid,
                "kind": "gate" if level == "FAIL" else "guardrail",
                "baseline": b,
                "candidate": c,
                "delta": None,
                "status": "flat",
            }
            rows.append(row)
            if not _finite(b):
                row["status"] = "no_baseline_value"
                continue
            if not _finite(c):
                row["status"] = "missing"
                if level == "REVIEW":
                    add(
                        "REVIEW",
                        "MISSING_GUARDRAIL_METRIC",
                        "present in baseline, absent in candidate",
                        mid,
                    )
                continue
            row["delta"] = c - b
            worse = _worsening(mid, b, c)
            if worse <= 0:
                row["status"] = "improved" if worse < 0 else "flat"
            elif worse <= _tolerance(policy, mid, b):
                row["status"] = "flat (within noise)"
            else:
                row["status"] = "REGRESSED"
                add(
                    level,
                    "REGRESSION",
                    f"{b:.3f} -> {c:.3f} (worse by {worse:.3f}, "
                    f"tolerance {_tolerance(policy, mid, b):.3f})",
                    mid,
                )

    # 6. slow-decay: gate metrics vs median of last N approved releases
    n = policy.regression.history_window
    if history and len(history) >= n:
        for mid in tp.gate_metrics:
            vals = [h.get("metrics", {}).get(mid) for h in history[-n:]]
            vals_f = [v for v in vals if _finite(v)]
            c = cm.get(mid)
            if len(vals_f) >= n and _finite(c):
                med = statistics.median(vals_f)
                worse = _worsening(mid, med, c)
                if worse > _tolerance(policy, mid, med):
                    add(
                        "FAIL",
                        "SLOW_DECAY",
                        f"{c:.3f} vs median of last {n} approved = {med:.3f}",
                        mid,
                    )

    if any(f.level == "FAIL" for f in findings):
        verdict: Literal["PASS", "REVIEW", "FAIL"] = "FAIL"
    elif any(f.level == "REVIEW" for f in findings):
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    return Verdict(verdict=verdict, tier=tier, findings=findings, rows=rows)


def format_report(v: Verdict, candidate: Snapshot | None = None) -> str:
    lines = ["=" * 88]
    if candidate:
        md = candidate.get("metadata", {})
        lines.append(
            f"candidate : {md.get('label') or '-'}  manifest={md.get('manifest_id', '?')} "
            f"dataset={md.get('dataset_version', '?')} policy={md.get('policy_version', '?')}"
        )
    lines.append(f"tier      : {v.tier}")
    lines.append("-" * 88)
    for f in sorted(v.findings, key=lambda x: {"FAIL": 0, "REVIEW": 1, "INFO": 2}[x.level]):
        where = f" [{f.metric}]" if f.metric else ""
        lines.append(f"{f.level:<6} {f.code}{where}: {f.message}")
    if v.rows:
        lines.append("-" * 88)
        lines.append(f"{'metric':<34}{'baseline':>10}{'candidate':>11}{'delta':>9}  status")
        for r in v.rows:

            def fmt(x: Any) -> str:
                return f"{x:.3f}" if _finite(x) else "-"

            lines.append(
                f"{r['metric']:<34}{fmt(r['baseline']):>10}{fmt(r['candidate']):>11}"
                f"{fmt(r['delta']):>9}  {r['status']}"
            )
    lines.append("=" * 88)
    lines.append(f"VERDICT: {v.verdict}")
    lines.append("=" * 88)
    return "\n".join(lines)

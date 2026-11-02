"""Turn per-case results into the flat metric ids that the policy and the gate key on.

Id scheme (same spirit as the reference repo, extended):
    <metric>.mean | .pass_rate | .wilson_lower | .n
    run.cases | run.errors | run.weighted_score
    ops.latency_p50_ms | ops.latency_p95_ms | ops.tokens_per_query | ops.success_rate
"""

from __future__ import annotations

import math
from collections import defaultdict

from mfg_rag_eval.models import CaseResult


def wilson_lower(passed: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval for a pass rate. Honest about small N."""
    if n == 0:
        return 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def aggregate(results: list[CaseResult], severity_weights: dict[str, float]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    passes: dict[str, int] = defaultdict(int)
    for r in results:
        for m in r.metrics:
            scores[m.name].append(m.score)
            passes[m.name] += int(m.passed)

    metrics: dict[str, float] = {}
    for name, vals in scores.items():
        n = len(vals)
        metrics[f"{name}.mean"] = sum(vals) / n
        metrics[f"{name}.pass_rate"] = passes[name] / n
        metrics[f"{name}.wilson_lower"] = wilson_lower(passes[name], n)
        metrics[f"{name}.n"] = float(n)

    ok = [r for r in results if r.error is None]
    metrics["run.cases"] = float(len(results))
    metrics["run.errors"] = float(len(results) - len(ok))

    # severity-weighted score: sum(w_i * s_i) / sum(w_i); an errored case scores 0.
    num = den = 0.0
    for r in results:
        w = severity_weights.get(r.severity, 1.0)
        s = (
            (sum(m.score for m in r.metrics) / len(r.metrics))
            if r.metrics and r.error is None
            else 0.0
        )
        num += w * s
        den += w
    metrics["run.weighted_score"] = num / den if den else 0.0

    lat = [r.output.latency_ms for r in ok if r.output]
    toks = [r.output.tokens_in + r.output.tokens_out for r in ok if r.output]
    metrics["ops.latency_p50_ms"] = percentile(lat, 50)
    metrics["ops.latency_p95_ms"] = percentile(lat, 95)
    metrics["ops.tokens_per_query"] = sum(toks) / len(toks) if toks else math.nan
    metrics["ops.success_rate"] = len(ok) / len(results) if results else 0.0
    return metrics

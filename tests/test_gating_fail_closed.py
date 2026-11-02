"""The most important tests in the repo: the release gate must fail CLOSED.

Each test pins a failure mode where the reference repo's compare.py would have passed.
"""

import copy
import math

import pytest

from mfg_rag_eval.gating import decide

GOOD = {
    "retrieval_recall.mean": 1.0,
    "citation_accuracy.pass_rate": 1.0,
    "revision_currency.pass_rate": 1.0,
    "identifier_fidelity.pass_rate": 1.0,
    "numeric_unit_fidelity.pass_rate": 1.0,
    "mandatory_warning.pass_rate": 1.0,
    "refusal_behavior.pass_rate": 1.0,
    "retrieval_mrr.mean": 1.0,
    "run.weighted_score": 1.0,
    "run.errors": 0.0,
    "ops.latency_p95_ms": 100.0,
}


def snap(**overrides):
    metrics = {**GOOD, **overrides}
    return {"metadata": {"tier": "T1"}, "metrics": metrics, "s1_critical_failures": []}


def codes(v):
    return {f.code for f in v.findings}


def test_clean_candidate_passes(policy):
    assert decide(policy, "T1", snap(), snap()).verdict == "PASS"


def test_missing_gate_metric_fails_not_passes(policy):
    cand = snap()
    del cand["metrics"]["numeric_unit_fidelity.pass_rate"]  # e.g. that eval crashed
    v = decide(policy, "T1", cand, snap())
    assert v.verdict == "FAIL" and "MISSING_GATE_METRIC" in codes(v)


@pytest.mark.parametrize("bad", [math.nan, None])
def test_nan_or_null_gate_metric_fails(policy, bad):
    v = decide(policy, "T1", snap(**{"mandatory_warning.pass_rate": bad}), snap())
    assert v.verdict == "FAIL" and "MISSING_GATE_METRIC" in codes(v)


def test_high_average_cannot_hide_an_s1_failure(policy):
    cand = snap()
    cand["s1_critical_failures"] = ["G-003:numeric_unit_fidelity"]
    v = decide(policy, "T1", cand, snap())
    assert v.verdict == "FAIL" and "S1_CRITICAL_FAILURE" in codes(v)


def test_run_errors_fail(policy):
    v = decide(policy, "T1", snap(**{"run.errors": 1.0}), snap())
    assert v.verdict == "FAIL" and "RUN_ERRORS" in codes(v)


def test_missing_run_errors_counter_is_malformed(policy):
    cand = snap()
    del cand["metrics"]["run.errors"]
    assert "MALFORMED_SNAPSHOT" in codes(decide(policy, "T1", cand, snap()))


def test_floor_enforced_even_without_baseline(policy):
    v = decide(policy, "T1", snap(**{"revision_currency.pass_rate": 0.9}), None)
    assert v.verdict == "FAIL" and "FLOOR_VIOLATION" in codes(v) and "NO_BASELINE" in codes(v)


def test_regression_beyond_tolerance_fails_within_tolerance_passes(policy):
    base = snap(**{"retrieval_recall.mean": 0.99})
    ok = decide(policy, "T1", snap(**{"retrieval_recall.mean": 0.97}), base)  # worse by 0.02 < 0.03
    bad = decide(policy, "T1", snap(**{"retrieval_recall.mean": 0.955}), base)  # worse by 0.035
    assert ok.verdict == "PASS"
    assert bad.verdict == "FAIL" and "REGRESSION" in codes(bad)


def test_improvement_is_never_penalised(policy):
    base = snap(**{"retrieval_recall.mean": 0.96})
    assert decide(policy, "T1", snap(), base).verdict == "PASS"


def test_guardrail_regression_is_review_not_fail(policy):
    v = decide(policy, "T1", snap(**{"retrieval_mrr.mean": 0.8}), snap())
    assert v.verdict == "REVIEW"


def test_latency_ceiling_breach_is_review(policy):
    v = decide(policy, "T1", snap(**{"ops.latency_p95_ms": 5000.0}), snap())
    assert v.verdict == "REVIEW" and "SLO_CEILING" in codes(v)


def test_fail_beats_review(policy):
    cand = snap(**{"retrieval_mrr.mean": 0.5, "revision_currency.pass_rate": 0.5})
    assert decide(policy, "T1", cand, snap()).verdict == "FAIL"


def test_slow_decay_caught_even_when_each_step_is_within_tolerance(policy):
    """Ratchet-down: baseline blessed at 0.99, then 0.97 each time. Last-3 median catches drift."""
    hist = [snap(**{"retrieval_recall.mean": v}) for v in (1.0, 1.0, 1.0)]
    base = snap(**{"retrieval_recall.mean": 0.98})
    cand = snap(**{"retrieval_recall.mean": 0.955})  # within tol of baseline, floor is 0.95
    assert decide(policy, "T1", cand, base).verdict == "PASS"
    v = decide(policy, "T1", cand, base, history=hist)
    assert v.verdict == "FAIL" and "SLOW_DECAY" in codes(v)


def test_decide_does_not_mutate_inputs(policy):
    cand, base = snap(), snap()
    before = copy.deepcopy((cand, base))
    decide(policy, "T1", cand, base)
    assert (cand, base) == before

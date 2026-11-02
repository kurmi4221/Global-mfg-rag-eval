# ADR 0002: The release gate fails closed

**Status:** Accepted and implemented (`gating/verdict.py`, `tests/test_gating_fail_closed.py`).

## Context
The reference harness this work builds on had three fail-open behaviours: a metric missing from
the candidate snapshot was labelled `dropped` and ignored; a NaN score was treated as `info`;
and safety gates keyed on averages, which the code itself notes can hide a single hard failure.
A crashed safety eval could therefore produce a PASS.

## Decision
- Missing, null or NaN gate metrics are FAIL.
- Errors during a run are FAIL.
- Any S1 case failing a tier-critical metric is FAIL regardless of aggregates.
- Absolute floors apply even with no baseline, and gate metrics are also compared to the median
  of the last N approved releases, so slow decay across several approvals is caught.
- Pass rates are reported with Wilson lower bounds so small-N results are not over-read.

## Consequences
Releases are blocked more often when infrastructure is flaky. That is intended: a blocked
release is recoverable, a wrongly approved safety regression may not be.

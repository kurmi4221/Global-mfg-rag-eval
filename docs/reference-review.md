# Review of the reference repo and what changed

Findings from reading `rag-eval-deepeval` (a course repo this work builds on), where each is
addressed here, and the test that pins the behaviour. Items marked "not done" are honest gaps.

| # | Finding in the reference code | Addressed in | Pinned by test |
|---|---|---|---|
| 1 | `compare.py` ignores a gate metric missing from the candidate (fails open) | `gating/verdict.py` | `test_missing_gate_metric_fails_not_passes` |
| 2 | NaN scores fall to `info` and are never gated | `verdict.py`, `runner.py` (NaN to null) | `test_nan_or_null_gate_metric_fails`, `test_snapshot_json_roundtrip_keeps_nan_as_null_and_fails_closed` |
| 3 | Safety gated on averages; worst case not gated | S1 critical-failure rule | `test_high_average_cannot_hide_an_s1_failure` |
| 4 | Regression-only comparison allows slow decay | floors + last-N median | `test_floor_enforced_even_without_baseline`, `test_slow_decay_caught_even_when_each_step_is_within_tolerance` |
| 5 | Online sampling used per-process `hash()`; `SAMPLE_RATE=1` sampled 100% | `online/sampler.py` | `test_stable_across_processes`, `test_rate_edges` |
| 6 | Online worker had no window, N+1 feedback calls, no idempotency | `online/worker.py` | `test_worker_is_idempotent_and_survives_restart`, `test_judge_failure_is_isolated_counted_and_retried` |
| 7 | Provenance gaps (retriever eval stamped `top_k: 3`, pipeline used 5) | pipeline `manifest()` + hashed manifest ID in every run | `test_manifest_is_complete_and_stable` |
| 8 | Snapshot kept only aggregates; runs not auditable | `ledger/` per-case hash chain | `test_ledger.py` |
| 9 | Retrieval scored only by an LLM judge | deterministic recall/MRR on gold `doc_id` + `revision` | `test_retrieval_recall_requires_revision_match` |
| 10 | Goldens lacked severity, evidence IDs, critical values | `models.Golden` + `goldens/schema.json` | `test_committed_json_schema_matches_the_pydantic_model` |
| 11 | Judge hardcoded to a cloud model in every eval | judge spec string, local (Ollama) supported | not exercised end to end |
| 12 | Pipeline re-invoked per eval | single run, outputs stored in ledger | `test_good_pipeline_has_no_critical_failures_and_a_valid_ledger` |

## Found while building this
The first version of the demo "good" pipeline cited a superseded revision for one S1 question
and the gate correctly blocked it. The fix was a revision filter at retrieval, the same
remediation the design recommends for stale-revision failures.

## Not done yet
- Noise calibration (10 unchanged runs, 2-sigma) is supported by per-metric tolerances in the
  policy but there is no calibration command yet; the default tolerance is a placeholder.
- Per-language slicing and judge calibration against human labels.
- Load testing; latency here is a single-user micro-benchmark on a toy corpus.
- End-to-end run against a real LangSmith instance and a real judge model.

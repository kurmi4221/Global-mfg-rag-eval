"""Offline lab runner: golden dataset -> pipeline -> metrics -> ledger -> snapshot.

One pipeline instance is built once and injected, so a snapshot is provably the measurement
of a single pipeline (the same principle as the reference repo's run_suite).
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mfg_rag_eval.adapters.pipeline import PipelineAdapter, manifest_id
from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.gating.aggregate import aggregate
from mfg_rag_eval.gating.policy import Policy
from mfg_rag_eval.ledger import RunLedger, file_sha256
from mfg_rag_eval.metrics import run_deterministic
from mfg_rag_eval.metrics.deepeval_judge import JudgeFn
from mfg_rag_eval.models import CaseResult, GoldenDataset


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_dataset(path: str | Path) -> GoldenDataset:
    return GoldenDataset.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def run_suite(
    *,
    pipeline: PipelineAdapter,
    dataset_path: str | Path,
    corpus: Corpus,
    policy: Policy,
    policy_path: str | Path,
    out_dir: str | Path,
    label: str = "",
    judge: JudgeFn | None = None,
    judge_pack: str = "none",
) -> tuple[dict[str, Any], list[CaseResult]]:
    t_start = time.perf_counter()
    dataset = load_dataset(dataset_path)
    tier_policy = policy.tiers[dataset.tier]
    manifest = pipeline.manifest()
    mid = manifest_id(manifest)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()  # one ledger per run directory
    ledger = RunLedger(ledger_path)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "git_sha": _git_sha(),
        "app_id": dataset.app_id,
        "tier": dataset.tier,
        "manifest_id": mid,
        "manifest": manifest,
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": file_sha256(dataset_path),
        "policy_version": policy.policy_version,
        "policy_sha256": file_sha256(policy_path),
        "corpus_snapshot_id": corpus.snapshot_id,
        "judge_pack": judge_pack,
    }
    ledger.append({"type": "run_start", **metadata})

    results: list[CaseResult] = []
    for golden in dataset.cases:
        result = CaseResult(
            case_id=golden.golden_id,
            severity=golden.severity,
            intent=golden.intent,
            language=golden.language,
            expected_behavior=golden.expected_behavior,
        )
        try:
            output = pipeline.invoke(golden.question)
            result.output = output
            result.metrics = run_deterministic(golden, output, corpus)
            if judge is not None:
                result.metrics += judge(golden, output)
            if golden.severity == "S1":
                result.critical_failures = [
                    m.name
                    for m in result.metrics
                    if not m.passed and m.name in tier_policy.critical_metrics
                ]
        except Exception as exc:  # fail closed: an error is never silently dropped
            result.error = f"{type(exc).__name__}: {exc}"
            if golden.severity == "S1":
                result.critical_failures = ["pipeline_error"]
        results.append(result)
        ledger.append({"type": "case", "run_manifest_id": mid, **result.model_dump(mode="json")})

    metrics = aggregate(results, policy.severity_weights)
    s1_critical = sorted(f"{r.case_id}:{name}" for r in results for name in r.critical_failures)
    metadata["suite_seconds"] = round(time.perf_counter() - t_start, 3)
    snapshot: dict[str, Any] = {
        "metadata": metadata,
        "metrics": {k: (None if v != v else v) for k, v in metrics.items()},  # NaN -> null in JSON
        "s1_critical_failures": s1_critical,
    }
    head = ledger.append(
        {"type": "run_end", "metrics": snapshot["metrics"], "s1_critical_failures": s1_critical}
    )
    snapshot["metadata"]["ledger_head"] = head
    (out / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    return snapshot, results

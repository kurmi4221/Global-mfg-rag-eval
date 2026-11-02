import json
from pathlib import Path

import pytest

from mfg_rag_eval import cli
from mfg_rag_eval.adapters import DemoPipeline
from mfg_rag_eval.gating import decide
from mfg_rag_eval.ledger import verify_ledger
from mfg_rag_eval.models import Golden
from mfg_rag_eval.runner import run_suite

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "goldens" / "mfg_v1.json"
POLICY = ROOT / "policies" / "tiers.yaml"


def run(corpus, policy, variant, out):
    return run_suite(
        pipeline=DemoPipeline(corpus, variant),
        dataset_path=DATASET,
        corpus=corpus,
        policy=policy,
        policy_path=POLICY,
        out_dir=out,
        label=variant,
    )


def test_good_pipeline_has_no_critical_failures_and_a_valid_ledger(corpus, policy, tmp_path):
    snap, results = run(corpus, policy, "good", tmp_path / "good")
    assert snap["s1_critical_failures"] == []
    assert snap["metrics"]["run.errors"] == 0
    assert verify_ledger(tmp_path / "good" / "ledger.jsonl")[0]
    assert (tmp_path / "good" / "snapshot.json").exists()
    # every case is in the ledger, plus run_start and run_end
    assert len((tmp_path / "good" / "ledger.jsonl").read_text().splitlines()) == len(results) + 2


@pytest.mark.parametrize("variant", ["stale", "nowarn", "broken"])
def test_each_regression_variant_is_blocked(corpus, policy, tmp_path, variant):
    base, _ = run(corpus, policy, "good", tmp_path / "base")
    cand, _ = run(corpus, policy, variant, tmp_path / variant)
    assert decide(policy, "T1", base, base).verdict == "PASS"
    assert decide(policy, "T1", cand, base).verdict == "FAIL"


def test_pipeline_exception_fails_closed(corpus, policy, tmp_path):
    class Exploding(DemoPipeline):
        def invoke(self, query):  # type: ignore[no-untyped-def]
            raise RuntimeError("vector store unreachable")

    snap, results = run_suite(
        pipeline=Exploding(corpus),
        dataset_path=DATASET,
        corpus=corpus,
        policy=policy,
        policy_path=POLICY,
        out_dir=tmp_path / "x",
    )
    assert snap["metrics"]["run.errors"] == len(results)
    assert decide(policy, "T1", snap, None).verdict == "FAIL"


def test_snapshot_json_roundtrip_keeps_nan_as_null_and_fails_closed(corpus, policy, tmp_path):
    snap, _ = run(corpus, policy, "good", tmp_path / "g")
    on_disk = json.loads((tmp_path / "g" / "snapshot.json").read_text())
    on_disk["metrics"]["numeric_unit_fidelity.pass_rate"] = None
    assert decide(policy, "T1", on_disk, snap).verdict == "FAIL"


def test_manifest_is_complete_and_stable(corpus):
    m1, m2 = DemoPipeline(corpus).manifest(), DemoPipeline(corpus).manifest()
    assert m1 == m2 and m1["corpus_snapshot_id"] == corpus.snapshot_id
    assert DemoPipeline(corpus, "broken").manifest() != m1


def test_cli_selftest_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert cli.main(["selftest"]) == 0


def test_cli_gate_exit_codes(tmp_path, monkeypatch, corpus, policy):
    monkeypatch.chdir(ROOT)
    base, _ = run(corpus, policy, "good", tmp_path / "b")
    bad, _ = run(corpus, policy, "broken", tmp_path / "c")
    b, c = tmp_path / "b" / "snapshot.json", tmp_path / "c" / "snapshot.json"
    assert cli.main(["gate", "--baseline", str(b), "--candidate", str(b)]) == 0
    assert cli.main(["gate", "--baseline", str(b), "--candidate", str(c)]) == 1
    assert base and bad


def test_cli_ledger_verify_and_online_demo(tmp_path, monkeypatch, corpus, policy):
    monkeypatch.chdir(ROOT)
    run(corpus, policy, "good", tmp_path / "g")
    assert cli.main(["ledger-verify", str(tmp_path / "g" / "ledger.jsonl")]) == 0
    assert cli.main(["online-demo"]) == 0


def test_all_goldens_are_valid_and_cover_required_behaviours():
    ds = json.loads(DATASET.read_text())
    cases = [Golden.model_validate(c) for c in ds["cases"]]
    assert len({c.golden_id for c in cases}) == len(cases)
    assert any(c.expected_behavior == "REFUSE" for c in cases)
    assert any(c.critical_values for c in cases) and any(c.identifiers for c in cases)
    assert any(c.severity == "S1" for c in cases)

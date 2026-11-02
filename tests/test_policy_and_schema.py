import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mfg_rag_eval.gating import load_policy
from mfg_rag_eval.models import GoldenDataset

ROOT = Path(__file__).resolve().parent.parent


def test_policy_loads_all_tiers_and_floors_reference_gate_metrics(policy):
    assert set(policy.tiers) == {"T1", "T2", "T3"}
    for name, tp in policy.tiers.items():
        assert tp.s1_failures_allowed == 0, name  # S1 tolerance is zero in every tier
        for mid in tp.floors:
            assert mid in tp.gate_metrics, f"{name}: floor on non-gate metric {mid}"


def test_policy_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text("policy_version: x\ntiers: {}\nsurprise: 1\n")
    with pytest.raises(ValidationError):
        load_policy(bad)


def test_committed_json_schema_matches_the_pydantic_model():
    committed = json.loads((ROOT / "goldens" / "schema.json").read_text())
    assert committed == GoldenDataset.model_json_schema(), (
        "goldens/schema.json is stale. Run: uv run python scripts/export_schema.py"
    )

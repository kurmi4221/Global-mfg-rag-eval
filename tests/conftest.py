from __future__ import annotations

from pathlib import Path

import pytest

from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.gating import Policy, load_policy
from mfg_rag_eval.models import Golden, PipelineOutput

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.load(ROOT / "data" / "synthetic_corpus.json")


@pytest.fixture(scope="session")
def policy() -> Policy:
    return load_policy(ROOT / "policies" / "tiers.yaml")


@pytest.fixture
def make_golden():
    def _make(**overrides) -> Golden:  # type: ignore[no-untyped-def]
        base = {
            "golden_id": "G-T",
            "question": "q",
            "severity": "S1",
            "intent": "t",
            "provenance": {"source": "synthetic", "author": "t"},
        }
        base.update(overrides)
        return Golden.model_validate(base)

    return _make


@pytest.fixture
def make_output():
    def _make(answer: str, citations=None, retrieved=None) -> PipelineOutput:  # type: ignore[no-untyped-def]
        return PipelineOutput.model_validate(
            {
                "query": "q",
                "answer": answer,
                "citations": citations or [],
                "retrieved": retrieved or [],
            }
        )

    return _make

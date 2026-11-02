"""Typed data contracts shared by the offline lab, the online floor and the ledger."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["T1", "T2", "T3"]
Severity = Literal["S1", "S2", "S3"]
ExpectedBehavior = Literal["ANSWER", "REFUSE", "CLARIFY", "ESCALATE"]

# The pipeline contract: an abstention is an exact, machine-checkable string.
ABSTAIN_TEXT = "I don't have enough information in the approved documents to answer that."


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------
# Golden dataset
# ----------------------------------------------------------------------------
class Evidence(_Strict):
    doc_id: str
    revision: str
    section: str | None = None
    page: int | None = None


class MustInclude(_Strict):
    kind: Literal["warning", "term"] = "warning"
    text: str


class CriticalValue(_Strict):
    kind: str
    value: str
    unit: str


class Provenance(_Strict):
    source: Literal["SME", "production", "synthetic"]
    author: str
    reviewer: str | None = None
    approved_at: str | None = None
    pii_scrubbed: bool = True


class Golden(_Strict):
    golden_id: str
    question: str
    severity: Severity
    intent: str
    expected_behavior: ExpectedBehavior = "ANSWER"
    language: str = "en"
    split: Literal["dev", "holdout"] = "dev"
    ideal_answer: str = ""
    gold_evidence: list[Evidence] = Field(default_factory=list)
    must_include: list[MustInclude] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    critical_values: list[CriticalValue] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    provenance: Provenance


class GoldenDataset(_Strict):
    dataset_version: str
    app_id: str
    tier: Tier
    cases: list[Golden]


# ----------------------------------------------------------------------------
# Corpus and pipeline I/O
# ----------------------------------------------------------------------------
class Chunk(_Strict):
    chunk_id: str
    doc_id: str
    revision: str
    section: str | None = None
    page: int | None = None
    text: str
    is_current: bool = True


class Citation(_Strict):
    doc_id: str
    revision: str
    section: str | None = None
    page: int | None = None


class RetrievedChunk(_Strict):
    chunk_id: str
    doc_id: str
    revision: str
    text: str
    score: float
    rank: int


class PipelineOutput(_Strict):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
class MetricResult(_Strict):
    name: str
    score: float
    passed: bool
    detail: str = ""
    method: Literal["deterministic", "judge"] = "deterministic"


class CaseResult(_Strict):
    case_id: str
    severity: Severity
    intent: str
    language: str
    expected_behavior: ExpectedBehavior
    error: str | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)
    output: PipelineOutput | None = None


# ----------------------------------------------------------------------------
# Online
# ----------------------------------------------------------------------------
class Trace(_Strict):
    run_id: str
    timestamp: datetime
    app_id: str
    tier: Tier
    query: str
    answer: str
    language: str = "en"
    intent: str = "unknown"
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    corpus_snapshot_id: str = ""
    pipeline_manifest_id: str = ""

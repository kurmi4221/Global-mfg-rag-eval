"""The pipeline contract and a self-contained demo implementation.

A plant implements `PipelineAdapter` (about 40 lines) around its own RAG stack. The platform
never owns the pipeline; it only calls `manifest()` and `invoke()`.

`DemoPipeline` is a deterministic BM25 + extractive generator so the whole platform
(gate, ledger, online worker) runs offline with no API keys. Its variants simulate realistic
regressions so the release gate has something real to catch:

    good    - correct behaviour
    stale   - revision filter disabled and the index prefers superseded revisions
    nowarn  - generator drops WARNING sentences
    broken  - both of the above
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from typing import Protocol, runtime_checkable

from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.models import (
    ABSTAIN_TEXT,
    Chunk,
    Citation,
    PipelineOutput,
    RetrievedChunk,
)
from mfg_rag_eval.textutil import tokenize

Manifest = dict[str, str | int | float | bool]

VARIANTS = ("good", "stale", "nowarn", "broken")


@runtime_checkable
class PipelineAdapter(Protocol):
    def manifest(self) -> Manifest:
        """Everything that defines this pipeline: model, prompt, embedding, chunker, top_k."""
        ...

    def invoke(self, query: str) -> PipelineOutput: ...


def manifest_id(manifest: Manifest) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class DemoPipeline:
    def __init__(
        self,
        corpus: Corpus,
        variant: str = "good",
        top_k: int = 3,
        min_coverage: float = 0.5,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.corpus = corpus
        self.variant = variant
        self.top_k = top_k
        self.min_coverage = min_coverage
        self.prefer_superseded = variant in ("stale", "broken")
        self.drop_warnings = variant in ("nowarn", "broken")
        self._doc_tokens = [tokenize(c.text) for c in corpus.chunks]
        self._avg_len = sum(len(t) for t in self._doc_tokens) / max(len(self._doc_tokens), 1)
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self._doc_tokens)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def manifest(self) -> Manifest:
        return {
            "pipeline": "demo-bm25-extractive",
            "variant": self.variant,
            "retriever": "bm25",
            "top_k": self.top_k,
            "min_coverage": self.min_coverage,
            "prefer_superseded": self.prefer_superseded,
            "drop_warnings": self.drop_warnings,
            "corpus_snapshot_id": self.corpus.snapshot_id,
        }

    # -- retrieval ---------------------------------------------------------
    def _bm25(self, q_tokens: list[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        toks = self._doc_tokens[idx]
        tf = Counter(toks)
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            denom = tf[t] + k1 * (1 - b + b * len(toks) / self._avg_len)
            score += self._idf.get(t, 0.0) * tf[t] * (k1 + 1) / denom
        return score

    def _coverage(self, q_tokens: list[str], idx: int) -> float:
        if not q_tokens:
            return 0.0
        doc = set(self._doc_tokens[idx])
        return sum(1 for t in set(q_tokens) if t in doc) / len(set(q_tokens))

    def _retrieve(self, query: str) -> list[tuple[Chunk, float, float]]:
        q = tokenize(query)
        scored: list[tuple[Chunk, float, float]] = []
        for i, chunk in enumerate(self.corpus.chunks):
            if not chunk.is_current and not self.prefer_superseded:
                continue  # revision filter: superseded documents are not retrievable
            s = self._bm25(q, i)
            if s <= 0:
                continue
            if self.prefer_superseded and not chunk.is_current:
                s *= 1.5  # simulated index bug: stale revisions outrank current ones
            scored.append((chunk, s, self._coverage(q, i)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]

    # -- generation --------------------------------------------------------
    def _answer(self, chunk: Chunk) -> str:
        text = chunk.text
        if self.drop_warnings:
            sentences = [s for s in text.split(". ") if not s.strip().startswith("WARNING:")]
            text = ". ".join(sentences)
        return text

    def invoke(self, query: str) -> PipelineOutput:
        t0 = time.perf_counter()
        hits = self._retrieve(query)
        retrieved = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                revision=c.revision,
                text=c.text,
                score=round(s, 4),
                rank=r,
            )
            for r, (c, s, _) in enumerate(hits, start=1)
        ]
        if not hits or hits[0][2] < self.min_coverage:
            answer, citations = ABSTAIN_TEXT, []
        else:
            top = hits[0][0]
            answer = self._answer(top)
            citations = [
                Citation(
                    doc_id=top.doc_id, revision=top.revision, section=top.section, page=top.page
                )
            ]
        latency_ms = (time.perf_counter() - t0) * 1000
        return PipelineOutput(
            query=query,
            answer=answer,
            citations=citations,
            retrieved=retrieved,
            latency_ms=round(latency_ms, 3),
            tokens_in=len(query.split()) + sum(len(r.text.split()) for r in retrieved),
            tokens_out=len(answer.split()),
        )

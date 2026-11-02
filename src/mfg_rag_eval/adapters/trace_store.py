"""Trace store adapters.

The online floor depends on the `TraceStore` interface, not on LangSmith. In regions where
LangSmith is available (EU SaaS, BYOC, self-hosted) use `LangSmithTraceStore`. In air-gapped
cells use `JsonlTraceStore` (or any OpenTelemetry-backed store implementing the same protocol).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from mfg_rag_eval.models import Citation, RetrievedChunk, Trace


class TraceStore(Protocol):
    def list_traces(self, since: datetime | None = None) -> list[Trace]: ...

    def write_trace(self, trace: Trace) -> None: ...

    def write_feedback(self, run_id: str, key: str, score: float, comment: str = "") -> None: ...


class JsonlTraceStore:
    """Local, dependency-free store. One JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.feedback_path = self.path.with_suffix(".feedback.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_trace(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def list_traces(self, since: datetime | None = None) -> list[Trace]:
        if not self.path.exists():
            return []
        out: list[Trace] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                trace = Trace.model_validate_json(line)
                if since is None or trace.timestamp >= since:
                    out.append(trace)
        return out

    def write_feedback(self, run_id: str, key: str, score: float, comment: str = "") -> None:
        rec = {"run_id": run_id, "key": key, "score": score, "comment": comment}
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class LangSmithTraceStore:
    """Reads root runs from a LangSmith project and writes judge scores back as feedback.

    Production apps emit traces themselves with `@traceable`; this adapter only reads them.
    For multi-geo deployments use one LangSmith instance per geography (set LANGSMITH_ENDPOINT).
    `client` is injectable so tests (and other SDK versions) can substitute a fake.
    """

    def __init__(self, project_name: str, client: Any | None = None) -> None:
        if client is None:
            try:
                from langsmith import Client
            except ImportError as e:  # pragma: no cover - depends on optional extra
                raise ImportError("Install the tracing extra: uv sync --extra tracing") from e
            client = Client()
        self.client = client
        self.project_name = project_name

    def write_trace(self, trace: Trace) -> None:
        raise NotImplementedError("Apps emit traces via @traceable; this store is read-only.")

    def list_traces(self, since: datetime | None = None) -> list[Trace]:
        kwargs: dict[str, Any] = {"project_name": self.project_name, "is_root": True}
        if since is not None:
            kwargs["start_time"] = since
        traces: list[Trace] = []
        for run in self.client.list_runs(**kwargs):
            trace = self._to_trace(run)
            if trace is not None:
                traces.append(trace)
        return traces

    def write_feedback(self, run_id: str, key: str, score: float, comment: str = "") -> None:
        self.client.create_feedback(run_id=run_id, key=key, score=score, comment=comment)

    @staticmethod
    def _to_trace(run: Any) -> Trace | None:
        inputs = getattr(run, "inputs", None) or {}
        outputs = getattr(run, "outputs", None) or {}
        extra = getattr(run, "extra", None) or {}
        meta = extra.get("metadata", {}) if isinstance(extra, dict) else {}
        query = inputs.get("query")
        answer = outputs.get("answer")
        if not query or answer is None:
            return None
        start = getattr(run, "start_time", None) or datetime.now(UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        return Trace(
            run_id=str(run.id),
            timestamp=start,
            app_id=meta.get("app_id", "unknown"),
            tier=meta.get("tier", "T3"),
            language=meta.get("language", "en"),
            intent=meta.get("intent", "unknown"),
            query=query,
            answer=answer,
            citations=[Citation.model_validate(c) for c in outputs.get("citations", [])],
            retrieved=[RetrievedChunk.model_validate(r) for r in outputs.get("retrieved", [])],
            latency_ms=float(outputs.get("latency_ms", 0.0)),
            corpus_snapshot_id=meta.get("corpus_snapshot_id", ""),
            pipeline_manifest_id=meta.get("pipeline_manifest_id", ""),
        )

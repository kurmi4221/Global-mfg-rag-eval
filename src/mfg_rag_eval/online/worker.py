"""Online worker: reads a time window of traces, applies L0 checks to all of them, runs an
optional LLM judge on a deterministic sample, and writes flags to a human review queue.

Design points (all fix problems in the reference eval_online.py):
  * time window + persisted watermark (with overlap) instead of re-listing everything
  * idempotency keys (run_id, check, judge_pack) so reprocessing is harmless
  * deterministic SHA-256 sampling, per-tier rates, 100% judging of anything L0 flags
  * state persisted atomically; the review queue never loses a flag on crash
  * judge failures are isolated per trace and counted, never silently dropped
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from mfg_rag_eval.adapters.trace_store import TraceStore
from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.models import Trace
from mfg_rag_eval.online.checks import CRITICAL_FLAGS, l0_checks
from mfg_rag_eval.online.sampler import sampled

OnlineJudge = Callable[[Trace], dict[str, float]]


@dataclass
class WorkerConfig:
    rates: dict[str, float] = field(default_factory=lambda: {"T1": 1.0, "T2": 0.25, "T3": 0.10})
    salt: str = "v1"
    overlap_seconds: int = 60
    judge_pack: str = "none"


@dataclass
class WindowReport:
    seen: int = 0
    skipped_already_done: int = 0
    flagged: int = 0
    critical: int = 0
    sampled: int = 0
    judged: int = 0
    judge_errors: int = 0


class OnlineWorker:
    def __init__(
        self,
        store: TraceStore,
        corpus: Corpus,
        state_path: str | Path,
        review_queue_path: str | Path,
        config: WorkerConfig | None = None,
        judge: OnlineJudge | None = None,
    ) -> None:
        self.store = store
        self.corpus = corpus
        self.state_path = Path(state_path)
        self.review_path = Path(review_queue_path)
        self.config = config or WorkerConfig()
        self.judge = judge
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        self._watermark: datetime | None = None
        self._done: set[str] = set()
        self._load_state()

    # -- state -------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        wm = data.get("watermark")
        self._watermark = datetime.fromisoformat(wm) if wm else None
        self._done = set(data.get("done", []))

    def _save_state(self) -> None:
        payload = {
            "watermark": self._watermark.isoformat() if self._watermark else None,
            "done": sorted(self._done),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.state_path)  # atomic on POSIX and Windows

    def _key(self, run_id: str, check: str) -> str:
        return f"{run_id}|{check}|{self.config.judge_pack}"

    # -- processing --------------------------------------------------------
    def process_window(self) -> WindowReport:
        report = WindowReport()
        since = (
            self._watermark - timedelta(seconds=self.config.overlap_seconds)
            if self._watermark
            else None
        )
        traces = sorted(self.store.list_traces(since), key=lambda t: t.timestamp)
        newest = self._watermark

        for trace in traces:
            report.seen += 1
            newest = trace.timestamp if newest is None or trace.timestamp > newest else newest
            l0_key = self._key(trace.run_id, "l0")
            flags: list[str] = []
            if l0_key in self._done:
                report.skipped_already_done += 1
            else:
                flags = l0_checks(trace, self.corpus)
                if flags:
                    self._done.add(self._key(trace.run_id, "flagged"))
                    report.flagged += 1
                    critical = sorted(f for f in flags if f in CRITICAL_FLAGS)
                    if critical:
                        report.critical += 1
                    self._enqueue(trace, flags, critical)
                self._done.add(l0_key)

            if self.judge is not None:
                judge_key = self._key(trace.run_id, "judge")
                rate = self.config.rates.get(trace.tier, 0.0)
                was_flagged = bool(flags) or self._key(trace.run_id, "flagged") in self._done
                pick = was_flagged or sampled(trace.run_id, rate, self.config.salt)
                if pick and judge_key not in self._done:
                    report.sampled += 1
                    try:
                        for key, score in self.judge(trace).items():
                            self.store.write_feedback(trace.run_id, key, score)
                        self._done.add(judge_key)
                        report.judged += 1
                    except Exception:
                        report.judge_errors += 1  # retried next window: key not marked done

        self._watermark = newest
        self._save_state()
        return report

    def _enqueue(self, trace: Trace, flags: list[str], critical: list[str]) -> None:
        item = {
            "run_id": trace.run_id,
            "timestamp": trace.timestamp.isoformat(),
            "app_id": trace.app_id,
            "tier": trace.tier,
            "flags": flags,
            "page_now": bool(critical),
            "query": trace.query,
            "answer": trace.answer,
        }
        with self.review_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

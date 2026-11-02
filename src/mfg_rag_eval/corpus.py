"""Revision-controlled corpus snapshot used by the demo pipeline and by citation checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mfg_rag_eval.models import Chunk


class Corpus:
    def __init__(self, chunks: list[Chunk], snapshot_id: str) -> None:
        self.chunks = chunks
        self.snapshot_id = snapshot_id
        self._by_doc_rev = {(c.doc_id, c.revision) for c in chunks}
        self._current = {(c.doc_id, c.revision) for c in chunks if c.is_current}

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        chunks = [Chunk.model_validate(c) for c in data["chunks"]]
        return cls(chunks, snapshot_id=hashlib.sha256(raw).hexdigest()[:12])

    def resolves(self, doc_id: str, revision: str) -> bool:
        """Does (doc_id, revision) exist in this corpus snapshot?"""
        return (doc_id, revision) in self._by_doc_rev

    def is_current(self, doc_id: str, revision: str) -> bool:
        return (doc_id, revision) in self._current

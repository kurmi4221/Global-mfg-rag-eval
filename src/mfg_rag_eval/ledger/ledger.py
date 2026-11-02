"""Append-only, hash-chained run ledger (JSONL).

Every line holds {seq, prev_hash, hash, record}, where hash = SHA-256(prev_hash + canonical
JSON of record). Editing, deleting or reordering any line breaks the chain, which `verify`
detects. In production, write the file to WORM-capable object storage and sign the final hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _link(prev_hash: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical(record)).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RunLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._prev = GENESIS
        if self.path.exists() and self.path.stat().st_size > 0:
            ok, msg = verify_ledger(self.path)
            if not ok:
                raise ValueError(f"refusing to append to a corrupt ledger: {msg}")
            last = self.path.read_text(encoding="utf-8").strip().splitlines()[-1]
            entry = json.loads(last)
            self._seq, self._prev = entry["seq"] + 1, entry["hash"]

    @property
    def head(self) -> str:
        return self._prev

    def append(self, record: dict[str, Any]) -> str:
        digest = _link(self._prev, record)
        entry = {"seq": self._seq, "prev_hash": self._prev, "hash": digest, "record": record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical(entry) + "\n")
        self._seq += 1
        self._prev = digest
        return digest


def verify_ledger(path: str | Path) -> tuple[bool, str]:
    prev = GENESIS
    p = Path(path)
    if not p.exists():
        return False, "ledger file not found"
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["seq"] != i:
            return False, f"line {i}: sequence broken (found {entry['seq']})"
        if entry["prev_hash"] != prev:
            return False, f"line {i}: prev_hash mismatch (chain broken)"
        if _link(prev, entry["record"]) != entry["hash"]:
            return False, f"line {i}: record was modified (hash mismatch)"
        prev = entry["hash"]
    return True, f"ok, head={prev[:16]}"

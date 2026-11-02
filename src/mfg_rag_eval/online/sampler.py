"""Deterministic sampling.

Fixes two bugs in the reference online worker:
  * `hash(str(run.id))` uses Python's per-process salted hash, so a restart (or a second
    worker) re-decides the sample. SHA-256 is stable across processes and machines.
  * `SAMPLE_RATE = 1` with `< SAMPLE_RATE * 100` silently sampled 100% while the comment said 30%.
    Here `rate` is a probability in [0, 1] and is validated.
"""

from __future__ import annotations

import hashlib


def sampled(run_id: str, rate: float, salt: str = "v1") -> bool:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if rate == 0.0:
        return False
    if rate == 1.0:
        return True
    digest = hashlib.sha256(f"{salt}:{run_id}".encode()).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < rate

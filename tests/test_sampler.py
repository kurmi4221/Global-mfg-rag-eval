import os
import subprocess
import sys

import pytest

from mfg_rag_eval.online.sampler import sampled


def test_deterministic_for_same_run_id():
    assert all(sampled("run-42", 0.3) == sampled("run-42", 0.3) for _ in range(20))


def test_stable_across_processes():
    """Python's built-in hash() is salted per process; ours must not be."""
    code = "from mfg_rag_eval.online.sampler import sampled; print(sampled('run-42', 0.5))"
    outs = {
        subprocess.check_output(
            [sys.executable, "-c", code], text=True, env={**os.environ, "PYTHONHASHSEED": str(seed)}
        ).strip()
        for seed in (1, 2, 3)
    }
    assert len(outs) == 1


def test_rate_edges():
    assert not any(sampled(f"r{i}", 0.0) for i in range(100))
    assert all(sampled(f"r{i}", 1.0) for i in range(100))


def test_rate_is_approximately_respected():
    n = 5000
    hits = sum(sampled(f"run-{i}", 0.3) for i in range(n))
    assert 0.27 * n < hits < 0.33 * n


def test_salt_changes_sample():
    a = {i for i in range(500) if sampled(f"r{i}", 0.3, "v1")}
    b = {i for i in range(500) if sampled(f"r{i}", 0.3, "v2")}
    assert a != b


@pytest.mark.parametrize("bad", [-0.1, 1.5, 30])
def test_invalid_rate_rejected(bad):
    with pytest.raises(ValueError):
        sampled("x", bad)

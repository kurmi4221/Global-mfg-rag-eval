from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from mfg_rag_eval.adapters import LangSmithTraceStore


class FakeClient:
    def __init__(self, runs):
        self.runs, self.list_kwargs, self.feedback = runs, None, []

    def list_runs(self, **kwargs):
        self.list_kwargs = kwargs
        return iter(self.runs)

    def create_feedback(self, **kwargs):
        self.feedback.append(kwargs)


def make_run(**over):
    base = {
        "id": "11111111-1111-1111-1111-111111111111",
        "inputs": {"query": "what is E-4471?"},
        "outputs": {
            "answer": "Low suction pressure.",
            "citations": [{"doc_id": "MAN-PUMP-110", "revision": "C"}],
            "retrieved": [],
            "latency_ms": 12.5,
        },
        "extra": {"metadata": {"app_id": "maint-assist-eu", "tier": "T1", "language": "de"}},
        "start_time": datetime(2026, 11, 2, 9, 0),  # naive on purpose
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_maps_root_runs_to_traces_and_uses_window():
    client = FakeClient([make_run(), make_run(inputs={}), make_run(outputs={})])
    store = LangSmithTraceStore("proj", client=client)
    since = datetime(2026, 11, 2, 8, 0, tzinfo=UTC)
    traces = store.list_traces(since)
    assert len(traces) == 1  # runs without query/answer are skipped, not crashed on
    t = traces[0]
    assert (t.app_id, t.tier, t.language) == ("maint-assist-eu", "T1", "de")
    assert t.timestamp.tzinfo is not None
    assert client.list_kwargs == {"project_name": "proj", "is_root": True, "start_time": since}


def test_feedback_written_back_and_traces_are_read_only():
    client = FakeClient([])
    store = LangSmithTraceStore("proj", client=client)
    store.write_feedback("rid", "faithfulness", 0.9, "ok")
    assert client.feedback == [
        {"run_id": "rid", "key": "faithfulness", "score": 0.9, "comment": "ok"}
    ]
    with pytest.raises(NotImplementedError):
        store.write_trace(None)  # type: ignore[arg-type]

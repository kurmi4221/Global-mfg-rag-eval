from datetime import UTC, datetime, timedelta

from mfg_rag_eval.adapters import DemoPipeline, JsonlTraceStore
from mfg_rag_eval.models import Citation, RetrievedChunk, Trace
from mfg_rag_eval.online import CRITICAL_FLAGS, OnlineWorker, WorkerConfig, l0_checks

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def trace(run_id, answer, citations=(), retrieved=(), tier="T1", offset=0):
    return Trace(
        run_id=run_id,
        timestamp=NOW + timedelta(seconds=offset),
        app_id="a",
        tier=tier,
        query="q",
        answer=answer,
        citations=list(citations),
        retrieved=list(retrieved),
    )


def chunk(corpus, chunk_id):
    c = next(c for c in corpus.chunks if c.chunk_id == chunk_id)
    return c, RetrievedChunk(
        chunk_id=c.chunk_id, doc_id=c.doc_id, revision=c.revision, text=c.text, score=1.0, rank=1
    )


def test_clean_trace_has_no_flags(corpus):
    c, r = chunk(corpus, "PUMP-C-9.3")
    t = trace("r1", c.text, [Citation(doc_id=c.doc_id, revision=c.revision)], [r])
    assert l0_checks(t, corpus) == []


def test_stale_invented_pii_and_citation_flags(corpus):
    c, r = chunk(corpus, "LOTO-E-4.2")
    stale = trace("r1", c.text, [Citation(doc_id=c.doc_id, revision="E")], [r])
    assert "STALE_REVISION" in l0_checks(stale, corpus)

    inv = trace("r2", "Use part 9999-000-Z", [Citation(doc_id="MAN-PUMP-110", revision="C")], [r])
    assert "INVENTED_IDENTIFIER" in l0_checks(inv, corpus)

    pii = trace(
        "r3",
        "Call +91 98765 43210 or a.b@plant.example",
        [Citation(doc_id="MAN-PUMP-110", revision="C")],
        [r],
    )
    assert "PII_IN_ANSWER" in l0_checks(pii, corpus)

    ghost = trace("r4", "ok", [Citation(doc_id="NOPE-1", revision="A")], [r])
    assert "UNRESOLVED_CITATION" in l0_checks(ghost, corpus)

    nocite = trace("r5", "A statement with no source", [], [r])
    assert "MISSING_CITATION" in l0_checks(nocite, corpus)


def test_abstention_is_not_flagged(corpus):
    pipe = DemoPipeline(corpus)
    out = pipe.invoke("What is the weather in Pune today?")
    t = trace("r1", out.answer, out.citations, out.retrieved)
    assert l0_checks(t, corpus) == []


def test_critical_flag_set_is_what_we_page_on():
    assert {"STALE_REVISION", "INVENTED_IDENTIFIER", "PII_IN_ANSWER"} == set(CRITICAL_FLAGS)


def make_worker(tmp_path, corpus, store, judge=None, rates=None):
    cfg = WorkerConfig(rates=rates or {"T1": 0.0, "T2": 0.0, "T3": 0.0}, judge_pack="jp-1")
    return OnlineWorker(
        store, corpus, tmp_path / "state.json", tmp_path / "queue.jsonl", cfg, judge
    )


def test_worker_is_idempotent_and_survives_restart(tmp_path, corpus):
    store = JsonlTraceStore(tmp_path / "t.jsonl")
    c, r = chunk(corpus, "LOTO-E-4.2")
    store.write_trace(trace("bad", c.text, [Citation(doc_id=c.doc_id, revision="E")], [r]))
    rep1 = make_worker(tmp_path, corpus, store).process_window()
    assert rep1.flagged == 1 and rep1.critical == 1
    rep2 = make_worker(tmp_path, corpus, store).process_window()  # new process, same state files
    assert rep2.flagged == 0 and rep2.skipped_already_done == 1
    assert len((tmp_path / "queue.jsonl").read_text().splitlines()) == 1


def test_worker_judges_sample_and_every_flagged_trace(tmp_path, corpus):
    store = JsonlTraceStore(tmp_path / "t.jsonl")
    c, r = chunk(corpus, "PUMP-C-9.3")
    ok_cite = [Citation(doc_id=c.doc_id, revision=c.revision)]
    for i in range(10):
        store.write_trace(trace(f"ok{i}", c.text, ok_cite, [r], offset=i))
    store.write_trace(trace("flagged", "Use part 9999-000-Z", ok_cite, [r], offset=20))

    judged: list[str] = []

    def judge(t: Trace) -> dict[str, float]:
        judged.append(t.run_id)
        return {"faithfulness": 0.9}

    rep = make_worker(
        tmp_path, corpus, store, judge, rates={"T1": 0.0, "T2": 0.0, "T3": 0.0}
    ).process_window()
    assert judged == ["flagged"] and rep.judged == 1  # rate 0 => only the flagged trace is judged
    assert (tmp_path / "t.feedback.jsonl").exists()


def test_judge_failure_is_isolated_counted_and_retried(tmp_path, corpus):
    store = JsonlTraceStore(tmp_path / "t.jsonl")
    c, r = chunk(corpus, "PUMP-C-9.3")
    store.write_trace(trace("x", c.text, [Citation(doc_id=c.doc_id, revision=c.revision)], [r]))
    calls = {"n": 0}

    def flaky(t: Trace) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("judge down")
        return {"faithfulness": 1.0}

    rates = {"T1": 1.0, "T2": 1.0, "T3": 1.0}
    w = make_worker(tmp_path, corpus, store, flaky, rates)
    assert w.process_window().judge_errors == 1
    assert make_worker(tmp_path, corpus, store, flaky, rates).process_window().judged == 1

"""Command-line interface.

mfg-rag-eval run      --variant good|stale|nowarn|broken
                      [--judge openai:<model>|ollama:<model>|grok:<model>]
mfg-rag-eval gate     --baseline A.json --candidate B.json
mfg-rag-eval demo     (baseline, a clean candidate, and a broken candidate)
mfg-rag-eval selftest (CI: exits 0 only if the gate passes good and blocks broken)
mfg-rag-eval online-demo
mfg-rag-eval ledger-verify runs/<name>/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mfg_rag_eval.adapters.pipeline import DemoPipeline, manifest_id
from mfg_rag_eval.adapters.trace_store import JsonlTraceStore
from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.gating import EXIT_CODE, Verdict, decide, format_report, load_policy
from mfg_rag_eval.ledger import verify_ledger
from mfg_rag_eval.models import Citation, RetrievedChunk, Trace
from mfg_rag_eval.online import OnlineWorker, WorkerConfig
from mfg_rag_eval.runner import load_dataset, run_suite

DEFAULT_DATASET = "goldens/mfg_v1.json"
DEFAULT_CORPUS = "data/synthetic_corpus.json"
DEFAULT_POLICY = "policies/tiers.yaml"


def _run(args: argparse.Namespace, variant: str, out: str, label: str) -> dict[str, Any]:
    corpus = Corpus.load(args.corpus)
    policy = load_policy(args.policy)
    pipeline = DemoPipeline(corpus, variant=variant)
    judge = None
    pack = "none"
    judge_spec = getattr(args, "judge", "none")
    if judge_spec and judge_spec != "none":
        from mfg_rag_eval.metrics.deepeval_judge import build_judge, judge_pack_id

        judge, pack = build_judge(judge_spec), judge_pack_id(judge_spec)
    snapshot, _ = run_suite(
        pipeline=pipeline,
        dataset_path=args.dataset,
        corpus=corpus,
        policy=policy,
        policy_path=args.policy,
        out_dir=out,
        label=label,
        judge=judge,
        judge_pack=pack,
    )
    return snapshot


def _gate(
    args: argparse.Namespace, baseline: dict[str, Any] | None, candidate: dict[str, Any]
) -> Verdict:
    policy = load_policy(args.policy)
    tier = candidate["metadata"]["tier"]
    verdict = decide(policy, tier, candidate, baseline)
    print(format_report(verdict, candidate))
    return verdict


def cmd_run(args: argparse.Namespace) -> int:
    out = args.out or f"runs/{args.variant}"
    snap = _run(args, args.variant, out, args.label or args.variant)
    print(f"snapshot -> {out}/snapshot.json   ledger -> {out}/ledger.jsonl")
    print(
        f"manifest={snap['metadata']['manifest_id']} metrics={len(snap['metrics'])} "
        f"s1_critical_failures={len(snap['s1_critical_failures'])}"
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    baseline = (
        json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    )
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    return EXIT_CODE[_gate(args, baseline, candidate).verdict]


def _demo_runs(args: argparse.Namespace) -> dict[str, Verdict]:
    shutil.rmtree("runs", ignore_errors=True)
    print("\n>>> 1/3  Bless the baseline: pipeline variant 'good'")
    baseline = _run(args, "good", "runs/baseline", "baseline (good)")
    print("\n>>> 2/3  Candidate = same pipeline, re-run (should PASS)")
    cand_ok = _run(args, "good", "runs/candidate_ok", "candidate: no change")
    v_ok = _gate(args, baseline, cand_ok)
    print("\n>>> 3/3  Candidate = bad index update (stale revisions) + generator drops warnings")
    cand_bad = _run(
        args, "broken", "runs/candidate_broken", "candidate: stale index + dropped warnings"
    )
    v_bad = _gate(args, baseline, cand_bad)
    return {"ok": v_ok, "broken": v_bad}


def cmd_demo(args: argparse.Namespace) -> int:
    _demo_runs(args)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    v = _demo_runs(args)
    good_ok = v["ok"].verdict == "PASS"
    bad_blocked = v["broken"].verdict == "FAIL"
    print(f"\nSELFTEST: clean candidate PASS={good_ok}  broken candidate blocked={bad_blocked}")
    return 0 if (good_ok and bad_blocked) else 1


def cmd_online_demo(args: argparse.Namespace) -> int:
    corpus = Corpus.load(args.corpus)
    dataset = load_dataset(args.dataset)
    shutil.rmtree("state", ignore_errors=True)
    store = JsonlTraceStore("state/traces.jsonl")
    pipeline = DemoPipeline(corpus, variant="good")
    mid = manifest_id(pipeline.manifest())
    base = datetime.now(UTC) - timedelta(minutes=5)
    for i, case in enumerate(dataset.cases):
        out = pipeline.invoke(case.question)
        store.write_trace(
            Trace(
                run_id=f"run-{i:03d}",
                timestamp=base + timedelta(seconds=i),
                app_id=dataset.app_id,
                tier=dataset.tier,
                intent=case.intent,
                query=out.query,
                answer=out.answer,
                citations=out.citations,
                retrieved=out.retrieved,
                latency_ms=out.latency_ms,
                corpus_snapshot_id=corpus.snapshot_id,
                pipeline_manifest_id=mid,
            )
        )
    # inject three faulty production traces
    chunk = corpus.chunks[1]  # superseded LOTO revision E
    retrieved = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            revision=chunk.revision,
            text=chunk.text,
            score=9.0,
            rank=1,
        )
    ]
    bad = [
        (
            "run-900",
            "Is the pump isolated?",
            chunk.text,
            [Citation(doc_id=chunk.doc_id, revision=chunk.revision)],
            "stale revision cited",
        ),
        (
            "run-901",
            "What is the part number?",
            "Use part 9999-000-Z. Contact j.doe@plant.example.",
            [Citation(doc_id="MAN-PUMP-110", revision="C")],
            "invented identifier + PII",
        ),
        (
            "run-902",
            "Where is the manual?",
            "See SOP-GHOST-0001.",
            [Citation(doc_id="SOP-GHOST-0001", revision="A")],
            "unresolvable citation",
        ),
    ]
    for j, (rid, q, a, cits, _) in enumerate(bad):
        store.write_trace(
            Trace(
                run_id=rid,
                timestamp=base + timedelta(seconds=60 + j),
                app_id=dataset.app_id,
                tier=dataset.tier,
                query=q,
                answer=a,
                citations=cits,
                retrieved=retrieved,
                corpus_snapshot_id=corpus.snapshot_id,
                pipeline_manifest_id=mid,
            )
        )

    worker = OnlineWorker(
        store,
        corpus,
        "state/worker_state.json",
        "state/review_queue.jsonl",
        WorkerConfig(rates={"T1": 0.0, "T2": 0.0, "T3": 0.0}),
    )
    print("window 1:", worker.process_window())
    print("window 2 (same data, nothing new - idempotent):", worker.process_window())
    print("\nreview queue (state/review_queue.jsonl):")
    for line in Path("state/review_queue.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        print(f"  {'PAGE' if item['page_now'] else 'ticket'}  {item['run_id']}  {item['flags']}")
    return 0


def cmd_ledger_verify(args: argparse.Namespace) -> int:
    ok, msg = verify_ledger(args.path)
    print(("VALID: " if ok else "INVALID: ") + msg)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mfg-rag-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", default=DEFAULT_DATASET)
        sp.add_argument("--corpus", default=DEFAULT_CORPUS)
        sp.add_argument("--policy", default=DEFAULT_POLICY)

    r = sub.add_parser("run", help="run the offline suite and write a snapshot + ledger")
    common(r)
    r.add_argument("--variant", default="good", choices=["good", "stale", "nowarn", "broken"])
    r.add_argument("--out", default=None)
    r.add_argument("--label", default="")
    r.add_argument(
        "--judge",
        default="none",
        help="none | openai:<model> | ollama:<model> | grok:<model>",
    )
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("gate", help="compare a candidate snapshot to a baseline; exit 0/1/2")
    g.add_argument("--policy", default=DEFAULT_POLICY)
    g.add_argument("--baseline", default=None)
    g.add_argument("--candidate", required=True)
    g.set_defaults(func=cmd_gate)

    for name, fn, helptext in (
        ("demo", cmd_demo, "baseline, clean candidate, broken candidate"),
        ("selftest", cmd_selftest, "CI check: gate must pass good and block broken"),
    ):
        d = sub.add_parser(name, help=helptext)
        common(d)
        d.set_defaults(func=fn, judge="none")

    o = sub.add_parser("online-demo", help="simulate production traces through the online worker")
    common(o)
    o.set_defaults(func=cmd_online_demo)

    v = sub.add_parser("ledger-verify", help="verify a hash-chained ledger file")
    v.add_argument("path")
    v.set_defaults(func=cmd_ledger_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

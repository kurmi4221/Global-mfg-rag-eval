"""Deterministic (no-LLM) metrics: fast, free, reproducible, auditable.

These are the PRIMARY signal for critical checks (units, identifiers, citations, refusals).
LLM judges are a second opinion layered on top; they never override a failure found here.

Each metric returns `None` when it does not apply to the case, so aggregates only include
cases where the metric is meaningful.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.models import CriticalValue, Golden, MetricResult, PipelineOutput
from mfg_rag_eval.textutil import extract_identifiers, is_abstention, normalize_text

MetricFn = Callable[[Golden, PipelineOutput, Corpus], MetricResult | None]


def _result(name: str, score: float, detail: str = "") -> MetricResult:
    return MetricResult(name=name, score=score, passed=score >= 1.0, detail=detail)


def _answerable(g: Golden) -> bool:
    return g.expected_behavior == "ANSWER"


# --------------------------------------------------------------------------- retrieval
def retrieval_recall(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g) or not g.gold_evidence:
        return None
    got = {(r.doc_id, r.revision) for r in out.retrieved}
    hit = [e for e in g.gold_evidence if (e.doc_id, e.revision) in got]
    missing = [f"{e.doc_id}@{e.revision}" for e in g.gold_evidence if e not in hit]
    return _result(
        "retrieval_recall",
        len(hit) / len(g.gold_evidence),
        f"missing: {missing}" if missing else "",
    )


def retrieval_mrr(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g) or not g.gold_evidence:
        return None
    gold = {(e.doc_id, e.revision) for e in g.gold_evidence}
    for r in out.retrieved:
        if (r.doc_id, r.revision) in gold:
            rr = 1.0 / r.rank
            return MetricResult(
                name="retrieval_mrr",
                score=rr,
                passed=rr >= 0.5,
                detail=f"first gold hit at rank {r.rank}",
            )
    return MetricResult(name="retrieval_mrr", score=0.0, passed=False, detail="no gold hit")


# --------------------------------------------------------------------------- citations
def citation_validity(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g):
        return None
    if not out.citations:
        return _result("citation_validity", 0.0, "answer has no citation")
    bad = [
        f"{c.doc_id}@{c.revision}"
        for c in out.citations
        if not corpus.resolves(c.doc_id, c.revision)
    ]
    return _result("citation_validity", 0.0 if bad else 1.0, f"unresolvable: {bad}" if bad else "")


def citation_accuracy(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g) or not g.gold_evidence:
        return None
    cited = {(c.doc_id, c.revision) for c in out.citations}
    hit = sum(1 for e in g.gold_evidence if (e.doc_id, e.revision) in cited)
    return _result(
        "citation_accuracy",
        hit / len(g.gold_evidence),
        f"cited={sorted(cited)}" if hit < len(g.gold_evidence) else "",
    )


def revision_currency(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g) or not out.citations:
        return None
    stale = [
        f"{c.doc_id}@{c.revision}"
        for c in out.citations
        if not corpus.is_current(c.doc_id, c.revision)
    ]
    return _result(
        "revision_currency",
        0.0 if stale else 1.0,
        f"superseded revision cited: {stale}" if stale else "",
    )


# --------------------------------------------------------------------------- identifiers
def identifier_fidelity(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    """Expected identifiers appear verbatim; none appear that are absent from the context."""
    if not _answerable(g):
        return None
    answer_ids = {i.lower() for i in extract_identifiers(out.answer)}
    context_ids = {i.lower() for r in out.retrieved for i in extract_identifiers(r.text)}
    expected = {i.lower() for i in g.identifiers}
    missing = sorted(expected - answer_ids)
    invented = sorted(answer_ids - context_ids)
    ok = not missing and not invented
    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if invented:
        parts.append(f"invented (not in retrieved context): {invented}")
    return _result("identifier_fidelity", 1.0 if ok else 0.0, "; ".join(parts))


# --------------------------------------------------------------------------- numbers + units
def normalize_numbers(text: str) -> str:
    """NFKC (full-width digits), thousands separators removed, decimal comma -> dot.

    Known ambiguity: '1,500' is treated as one thousand five hundred (thousands separator).
    A German '1,500' meaning 1.5 would be misread; plants with such data should add explicit
    locale handling (see docs/adr/0003).
    """
    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", s)
    return s


def unit_variants(unit: str) -> set[str]:
    u = unicodedata.normalize("NFKC", unit).lower()
    return {u, u.replace("*", "\u00b7"), u.replace("*", ""), u.replace("*", " ")}


_NUM_UNIT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*([a-z\u00b0\u00b5%*\u00b7/-]+(?: [a-z]+)?)?"
)


def _to_decimal(s: str) -> Decimal | None:
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _value_present(answer_norm: str, cv: CriticalValue) -> tuple[bool, list[str]]:
    """(found, wrong_unit_notes) for one critical value."""
    want = _to_decimal(normalize_numbers(cv.value))
    variants = unit_variants(cv.unit)
    wrong: list[str] = []
    for m in _NUM_UNIT_RE.finditer(answer_norm):
        got = _to_decimal(m.group(1))
        if got is None or want is None or got != want:
            continue
        unit_tok = (m.group(2) or "").strip()
        # accept if the unit token starts with any accepted spelling (handles "hours of operation")
        if any(unit_tok == v or unit_tok.startswith(v + " ") for v in variants):
            return True, []
        wrong.append(f"{m.group(1)} {unit_tok or '(no unit)'}")
    return False, wrong


def numeric_unit_fidelity(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not _answerable(g) or not g.critical_values:
        return None
    answer_norm = normalize_numbers(normalize_text(out.answer))
    found, notes = 0, []
    for cv in g.critical_values:
        ok, wrong = _value_present(answer_norm, cv)
        if ok:
            found += 1
        elif wrong:
            notes.append(f"{cv.kind}: expected {cv.value} {cv.unit}, got {wrong}")
        else:
            notes.append(f"{cv.kind}: {cv.value} {cv.unit} not found")
    return _result("numeric_unit_fidelity", found / len(g.critical_values), "; ".join(notes))


# --------------------------------------------------------------------------- content rules
def mandatory_warning(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    items = [m for m in g.must_include if m.kind == "warning"]
    if not _answerable(g) or not items:
        return None
    ans = normalize_text(out.answer)
    missing = [m.text for m in items if normalize_text(m.text) not in ans]
    return _result(
        "mandatory_warning",
        (len(items) - len(missing)) / len(items),
        f"warning content missing: {missing}" if missing else "",
    )


def required_content(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    items = [m for m in g.must_include if m.kind == "term"]
    if not _answerable(g) or not items:
        return None
    ans = normalize_text(out.answer)
    missing = [m.text for m in items if normalize_text(m.text) not in ans]
    return _result(
        "required_content",
        (len(items) - len(missing)) / len(items),
        f"missing terms: {missing}" if missing else "",
    )


def forbidden_content(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    if not g.must_not_include:
        return None
    ans = normalize_text(out.answer)
    present = [t for t in g.must_not_include if normalize_text(t) in ans]
    return _result(
        "forbidden_content",
        0.0 if present else 1.0,
        f"forbidden text present: {present}" if present else "",
    )


def refusal_behavior(g: Golden, out: PipelineOutput, corpus: Corpus) -> MetricResult | None:
    abstained = is_abstention(out.answer)
    if g.expected_behavior == "REFUSE":
        return _result(
            "refusal_behavior",
            1.0 if abstained else 0.0,
            "" if abstained else "answered a question that must be refused",
        )
    if g.expected_behavior == "ANSWER":
        return _result(
            "refusal_behavior", 0.0 if abstained else 1.0, "over-refusal" if abstained else ""
        )
    return None


DETERMINISTIC_METRICS: list[MetricFn] = [
    retrieval_recall,
    retrieval_mrr,
    citation_validity,
    citation_accuracy,
    revision_currency,
    identifier_fidelity,
    numeric_unit_fidelity,
    mandatory_warning,
    required_content,
    forbidden_content,
    refusal_behavior,
]


def run_deterministic(g: Golden, out: PipelineOutput, corpus: Corpus) -> list[MetricResult]:
    results = (fn(g, out, corpus) for fn in DETERMINISTIC_METRICS)
    return [r for r in results if r is not None]

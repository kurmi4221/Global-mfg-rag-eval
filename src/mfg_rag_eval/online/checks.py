"""L0 online checks: deterministic, run on 100% of traffic, near-zero cost.

Only reference-free checks can run online (there is no answer key in production).
"""

from __future__ import annotations

from mfg_rag_eval.corpus import Corpus
from mfg_rag_eval.models import Trace
from mfg_rag_eval.textutil import extract_identifiers, find_pii, is_abstention

# Flags that should page a human immediately (S1-class events).
CRITICAL_FLAGS = frozenset({"STALE_REVISION", "INVENTED_IDENTIFIER", "PII_IN_ANSWER"})


def l0_checks(trace: Trace, corpus: Corpus) -> list[str]:
    flags: list[str] = []
    abstained = is_abstention(trace.answer)

    if not trace.retrieved and not abstained:
        flags.append("EMPTY_RETRIEVAL")
    if not abstained and not trace.citations:
        flags.append("MISSING_CITATION")
    for c in trace.citations:
        if not corpus.resolves(c.doc_id, c.revision):
            flags.append("UNRESOLVED_CITATION")
            break
    if any(
        corpus.resolves(c.doc_id, c.revision) and not corpus.is_current(c.doc_id, c.revision)
        for c in trace.citations
    ):
        flags.append("STALE_REVISION")

    context_ids = {i.lower() for r in trace.retrieved for i in extract_identifiers(r.text)}
    answer_ids = {i.lower() for i in extract_identifiers(trace.answer)}
    if answer_ids - context_ids:
        flags.append("INVENTED_IDENTIFIER")
    if find_pii(trace.answer):
        flags.append("PII_IN_ANSWER")
    return flags

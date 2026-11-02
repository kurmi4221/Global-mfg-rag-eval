from mfg_rag_eval.metrics.deterministic import (
    citation_validity,
    identifier_fidelity,
    mandatory_warning,
    normalize_numbers,
    numeric_unit_fidelity,
    refusal_behavior,
    retrieval_recall,
    revision_currency,
)
from mfg_rag_eval.models import ABSTAIN_TEXT


def cv(value, unit, kind="torque"):
    return {"kind": kind, "value": value, "unit": unit}


# ---- numbers and units -------------------------------------------------------
def test_normalize_decimal_comma_and_fullwidth_digits():
    assert normalize_numbers("1,5 Nm") == "1.5 Nm"
    assert normalize_numbers("25,00 mm") == "25.00 mm"
    assert normalize_numbers("\uff14\uff15 N*m") == "45 N*m"  # full-width 45
    assert normalize_numbers("1,500 hours") == "1500 hours"


def test_numeric_pass_with_unit_spellings(corpus, make_golden, make_output):
    g = make_golden(critical_values=[cv("45", "N*m")])
    for ans in ("Tighten to 45 N*m.", "Use 45 Nm in a star pattern", "45 N\u00b7m"):
        r = numeric_unit_fidelity(g, make_output(ans), corpus)
        assert r is not None and r.passed, ans


def test_numeric_fails_on_wrong_value_and_wrong_unit(corpus, make_golden, make_output):
    g = make_golden(critical_values=[cv("45", "N*m")])
    wrong_value = numeric_unit_fidelity(g, make_output("Tighten to 54 N*m"), corpus)
    wrong_unit = numeric_unit_fidelity(g, make_output("Tighten to 45 lbf-ft"), corpus)
    assert wrong_value is not None and not wrong_value.passed
    assert wrong_unit is not None and not wrong_unit.passed
    assert "lbf-ft" in wrong_unit.detail


def test_numeric_decimal_comma_equivalent(corpus, make_golden, make_output):
    g = make_golden(critical_values=[cv("25.00", "mm", "dimension")])
    r = numeric_unit_fidelity(g, make_output("Der Durchmesser ist 25,00 mm"), corpus)
    assert r is not None and r.passed


def test_numeric_does_not_match_substring_of_larger_number(corpus, make_golden, make_output):
    g = make_golden(critical_values=[cv("45", "N*m")])
    r = numeric_unit_fidelity(g, make_output("Tighten to 145 N*m"), corpus)
    assert r is not None and not r.passed


# ---- identifiers -------------------------------------------------------------
def test_identifier_invented_is_caught(corpus, make_golden, make_output):
    g = make_golden(identifiers=["7740-221-B"])
    retrieved = [
        {
            "chunk_id": "c",
            "doc_id": "d",
            "revision": "A",
            "text": "Part 7740-221-B fits.",
            "score": 1.0,
            "rank": 1,
        }
    ]
    ok = identifier_fidelity(g, make_output("Use 7740-221-B", retrieved=retrieved), corpus)
    transposed = identifier_fidelity(g, make_output("Use 7740-212-B", retrieved=retrieved), corpus)
    assert ok is not None and ok.passed
    assert transposed is not None and not transposed.passed
    assert "invented" in transposed.detail and "missing" in transposed.detail


# ---- citations and revisions -------------------------------------------------
def test_citation_validity_and_revision_currency(corpus, make_golden, make_output):
    g = make_golden()
    current = [{"doc_id": "SOP-LOTO-0042", "revision": "F"}]
    stale = [{"doc_id": "SOP-LOTO-0042", "revision": "E"}]
    ghost = [{"doc_id": "SOP-NOPE-1", "revision": "A"}]
    assert revision_currency(g, make_output("a", current), corpus).passed  # type: ignore[union-attr]
    assert not revision_currency(g, make_output("a", stale), corpus).passed  # type: ignore[union-attr]
    assert citation_validity(g, make_output("a", stale), corpus).passed  # type: ignore[union-attr]
    assert not citation_validity(g, make_output("a", ghost), corpus).passed  # type: ignore[union-attr]
    assert not citation_validity(g, make_output("a", []), corpus).passed  # type: ignore[union-attr]


def test_retrieval_recall_requires_revision_match(corpus, make_golden, make_output):
    g = make_golden(gold_evidence=[{"doc_id": "SOP-LOTO-0042", "revision": "F"}])
    only_old = [
        {
            "chunk_id": "c",
            "doc_id": "SOP-LOTO-0042",
            "revision": "E",
            "text": "x",
            "score": 1.0,
            "rank": 1,
        }
    ]
    r = retrieval_recall(g, make_output("a", retrieved=only_old), corpus)
    assert r is not None and r.score == 0.0


# ---- warnings and refusals ---------------------------------------------------
def test_mandatory_warning_missing_fails(corpus, make_golden, make_output):
    g = make_golden(must_include=[{"kind": "warning", "text": "zero energy state"}])
    assert mandatory_warning(g, make_output("Verify the Zero Energy State first"), corpus).passed  # type: ignore[union-attr]
    assert not mandatory_warning(g, make_output("Apply lockout tagout."), corpus).passed  # type: ignore[union-attr]


def test_refusal_both_directions(corpus, make_golden, make_output):
    refuse = make_golden(expected_behavior="REFUSE")
    answer = make_golden(expected_behavior="ANSWER")
    assert refusal_behavior(refuse, make_output(ABSTAIN_TEXT), corpus).passed  # type: ignore[union-attr]
    assert not refusal_behavior(refuse, make_output("The torque is 12 N*m"), corpus).passed  # type: ignore[union-attr]
    assert not refusal_behavior(answer, make_output(ABSTAIN_TEXT), corpus).passed  # type: ignore[union-attr]
    assert refusal_behavior(answer, make_output("It is 12 N*m"), corpus).passed  # type: ignore[union-attr]


def test_metrics_skip_when_not_applicable(corpus, make_golden, make_output):
    g = make_golden(expected_behavior="REFUSE")
    assert retrieval_recall(g, make_output(ABSTAIN_TEXT), corpus) is None
    assert numeric_unit_fidelity(g, make_output(ABSTAIN_TEXT), corpus) is None

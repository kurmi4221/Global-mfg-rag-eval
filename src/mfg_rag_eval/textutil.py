"""Small, dependency-free text helpers shared by the pipeline, metrics and online checks."""

from __future__ import annotations

import re
import unicodedata

from mfg_rag_eval.models import ABSTAIN_TEXT

_STOPWORD_TEXT = (
    "a an the and or of to in on at for with by from is are was were be been do does did can "
    "could should would must may might what which who whom how why when where i me my we our "
    "you your it its this that these those as about before after into over under use using "
    "tell please there their then than not no yes if"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-*/.][a-z0-9]+)*")
_IDENT_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<![\w-])\+?\d(?:[\s().-]?\d){9,}(?![\w-])")


def stem(token: str) -> str:
    """Tiny plural stripper: good enough for the POC retriever, intentionally simple."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_text(text: str) -> str:
    """NFKC (folds full-width digits/letters), lower-case, collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize_text(text))
    tokens = [stem(t) for t in tokens]
    return [t for t in tokens if not (drop_stopwords and t in STOPWORDS)]


def extract_identifiers(text: str) -> list[str]:
    """Hyphenated tokens containing a digit: part numbers, error codes, doc IDs (E-4471, P-110)."""
    normalized = unicodedata.normalize("NFKC", text)
    found = [
        m.group(0) for m in _IDENT_RE.finditer(normalized) if any(c.isdigit() for c in m.group(0))
    ]
    return sorted(set(found), key=str.lower)


def is_abstention(answer: str) -> bool:
    return normalize_text(answer) == normalize_text(ABSTAIN_TEXT)


def find_pii(text: str) -> list[str]:
    hits = [m.group(0) for m in _EMAIL_RE.finditer(text)]
    hits += [m.group(0) for m in _PHONE_RE.finditer(text)]
    return hits

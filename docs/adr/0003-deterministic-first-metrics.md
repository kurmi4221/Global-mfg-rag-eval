# ADR 0003: Deterministic checks are primary; LLM judges are a second opinion

**Status:** Accepted and implemented.

## Context
Manufacturing answers hinge on exact tokens: part numbers, error codes, revision letters,
torque values and units. An LLM judge can score a wrong digit as "mostly faithful".

## Decision
Critical metrics are computed by code: gold-evidence recall, citation validity and accuracy,
revision currency, identifier fidelity (including invented identifiers), numeric and unit
fidelity, mandatory warnings and refusal behaviour. DeepEval judge metrics (faithfulness, answer
relevancy) are guardrails (REVIEW), never the sole basis for approving a safety-critical answer.

## Known limitation
Number normalization treats `1,500` as one thousand five hundred. A locale where that string
means 1.5 would be misread. Plants with such data need explicit per-locale handling before
numeric checks are trusted there.

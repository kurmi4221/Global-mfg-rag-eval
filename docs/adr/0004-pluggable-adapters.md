# ADR 0004: Pipelines, trace stores and judges sit behind adapters

**Status:** Accepted and implemented.

## Decision
- `PipelineAdapter`: plants wrap their own RAG stack (`manifest()` + `invoke()`).
- `TraceStore`: LangSmith where available (EU SaaS, BYOC, self-hosted); JSONL or an
  OpenTelemetry-backed store for air-gapped sites.
- Judges: `openai:<model>` or `ollama:<model>` specs through DeepEval; any `DeepEvalBaseLLM`
  can be added.

## Consequences
The platform does not depend on a single vendor, can run offline, and is testable without keys.
The LangSmith adapter is verified against a fake client, not a live LangSmith instance.

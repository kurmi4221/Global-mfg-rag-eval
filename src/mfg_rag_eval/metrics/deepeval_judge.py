"""Optional LLM-as-judge metrics via DeepEval (faithfulness, answer relevancy).

Imports are lazy so the platform, the gate and the test-suite run without DeepEval or any API
key. Install with `uv sync --extra judge`.

Judge spec strings:
    openai:gpt-4o-mini        cloud judge (only where policy permits)
    ollama:llama3.1          local judge for restricted / air-gapped cells
    grok:grok-beta           xAI Grok cloud judge (OpenAI-compatible endpoint)

A judge pack is the judge model + DeepEval version + thresholds. Scores are comparable only
within one judge-pack version, so the pack string is stamped into every run.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mfg_rag_eval.models import Golden, MetricResult, PipelineOutput
from mfg_rag_eval.textutil import is_abstention

JudgeFn = Callable[[Golden, PipelineOutput], list[MetricResult]]


def judge_pack_id(spec: str) -> str:
    try:
        version = metadata.version("deepeval")
    except metadata.PackageNotFoundError:
        version = "not-installed"
    return f"{spec}|deepeval-{version}"


def _resolve_model(spec: str) -> Any:
    provider, _, name = spec.partition(":")
    if provider == "openai":
        return name or "gpt-4o-mini"
    if provider == "ollama":
        from deepeval.models import OllamaModel

        return OllamaModel(model=name or "llama3.1")
    if provider in ("grok", "xai"):
        from deepeval.models import OpenAIModel

        load_dotenv(Path.cwd() / ".env")
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError(
                "Grok judge key not found. Copy .env.example to .env, set XAI_API_KEY or "
                "GROK_API_KEY, then rerun the command."
            )
        # xAI uses an OpenAI-compatible API endpoint (https://api.x.ai/v1)
        return OpenAIModel(
            model=name or "grok-beta",
            base_url="https://api.x.ai/v1",
            api_key=api_key,
        )
    raise ValueError(
        f"Unknown judge spec {spec!r}; use openai:<model>, ollama:<model>, or grok:<model>"
    )


def build_judge(spec: str, threshold: float = 0.7) -> JudgeFn:
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        raise ImportError("Judge metrics need DeepEval: uv sync --extra judge") from e

    model = _resolve_model(spec)

    def judge(golden: Golden, out: PipelineOutput) -> list[MetricResult]:
        if golden.expected_behavior != "ANSWER" or is_abstention(out.answer):
            return []
        context: Any = [r.text for r in out.retrieved]
        case = LLMTestCase(input=out.query, actual_output=out.answer, retrieval_context=context)
        results: list[MetricResult] = []
        for name, metric in (
            (
                "faithfulness",
                FaithfulnessMetric(threshold=threshold, model=model, include_reason=True),
            ),
            (
                "answer_relevancy",
                AnswerRelevancyMetric(threshold=threshold, model=model, include_reason=True),
            ),
        ):
            metric.measure(case)
            results.append(
                MetricResult(
                    name=name,
                    score=float(metric.score or 0.0),
                    passed=bool(metric.is_successful()),
                    detail=str(getattr(metric, "reason", "") or ""),
                    method="judge",
                )
            )
        return results

    return judge

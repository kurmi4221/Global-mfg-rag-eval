import sys
from types import ModuleType

import pytest

from mfg_rag_eval.metrics.deepeval_judge import _resolve_model


def test_grok_model_uses_key_from_dotenv(tmp_path, monkeypatch):
    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    deepeval_module = ModuleType("deepeval")
    deepeval_module.__path__ = []
    models_module = ModuleType("deepeval.models")
    models_module.OpenAIModel = FakeOpenAIModel
    monkeypatch.setitem(sys.modules, "deepeval", deepeval_module)
    monkeypatch.setitem(sys.modules, "deepeval.models", models_module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    (tmp_path / ".env").write_text("XAI_API_KEY=test-xai-key\n", encoding="utf-8")

    model = _resolve_model("grok:grok-2-latest")

    assert model.kwargs == {
        "model": "grok-2-latest",
        "base_url": "https://api.x.ai/v1",
        "api_key": "test-xai-key",
    }


def test_grok_model_requires_an_xai_api_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr("mfg_rag_eval.metrics.deepeval_judge.load_dotenv", lambda _: False)

    with pytest.raises(ValueError, match="XAI_API_KEY or GROK_API_KEY"):
        _resolve_model("grok:grok-beta")

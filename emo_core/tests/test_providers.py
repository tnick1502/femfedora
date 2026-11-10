import httpx
import pytest

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.providers.llm_provider import OpenAICompatibleProvider


async def test_llm_http_contract(config, monkeypatch):
    """Проверяет URL, модель, заголовок авторизации и формат OpenAI-совместимого запроса."""
    config.llm_base_url = "https://llm.test/v1"
    config.llm_model = "test-model"
    config.llm_api_key = "test-key"
    original = httpx.AsyncClient

    def handler(request):
        """Проверяет HTTP-запрос и возвращает синтетический ответ LLM без внешней сети."""
        import json

        data = json.loads(request.content)
        assert str(request.url) == "https://llm.test/v1/chat/completions"
        assert data["model"] == "test-model" and data["messages"][0]["role"] == "system"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(**kw, transport=httpx.MockTransport(handler))
    )
    assert (
        await OpenAICompatibleProvider(config).generate([{"role": "system", "content": "rules"}])
        == "Ответ"
    )


async def test_llm_api_failure_sanitized(config, monkeypatch):
    """Проверяет замену тела ошибки провайдера фиксированным кодом llm_unavailable."""
    original = httpx.AsyncClient
    config.llm_base_url = "https://llm.test/v1"
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            **kw,
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, text="sensitive provider body")
            ),
        ),
    )
    with pytest.raises(ProviderUnavailable) as exc:
        await OpenAICompatibleProvider(config).generate([])
    assert str(exc.value) == "llm_unavailable"

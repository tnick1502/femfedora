import httpx
import pytest

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.providers.llm_provider import OpenAICompatibleProvider


async def test_llm_http_contract(config, monkeypatch):
    """Проверяет URL, модель, заголовок авторизации и формат OpenAI-совместимого запроса."""
    config.llm_base_url = "https://llm.test/v1"
    config.llm_model = "test-model"
    config.llm_api_key = "test-key"
    config.llm_extra_body = {
        "model": "must-not-override",
        "messages": [],
        "temperature": 2,
        "max_tokens": 1,
        "custom_option": "preserved",
    }
    original = httpx.AsyncClient

    def handler(request):
        """Проверяет HTTP-запрос и возвращает синтетический ответ LLM без внешней сети."""
        import json

        data = json.loads(request.content)
        assert str(request.url) == "https://llm.test/v1/chat/completions"
        assert data["model"] == "test-model"
        assert data["messages"] == [{"role": "system", "content": "rules"}]
        assert data["temperature"] == 0.4 and data["max_tokens"] == config.llm_max_output_tokens
        assert data["custom_option"] == "preserved"
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


async def test_large_streamed_llm_response_is_stopped_and_sanitized(config, monkeypatch):
    consumed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                consumed.append(True)
                yield b"x" * 65536

    original = httpx.AsyncClient
    config.llm_base_url = "https://llm.test/v1"
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            **kw, transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Body()))
        ),
    )
    with pytest.raises(ProviderUnavailable, match="llm_unavailable"):
        await OpenAICompatibleProvider(config).generate([])
    assert len(consumed) == 33  # stop just beyond 2 MiB rather than consuming the whole stream

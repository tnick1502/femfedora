from typing import Protocol

import httpx

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.core.http import json_request


class LLMText(str):
    """Текст ответа с фактическим расходом токенов провайдера (если он его сообщил)."""

    usage: dict | None = None


class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], repair: bool = False) -> str:
        """Возвращает текст ответа на последовательность сообщений role/content.

        repair=True запрашивает исправленную генерацию после неудачной проверки.
        Реализация сообщает недоступность через ProviderUnavailable.
        """
        ...


def generation_messages(messages, repair=False):
    if not repair:
        return list(messages)
    return [
        *messages[:1],
        {
            "role": "system",
            "content": "Предыдущий ответ нарушил правила. Дай короткую безопасную поддержку без диагнозов, лечения и медицинских гарантий.",
        },
        *messages[1:],
    ]


class OpenAICompatibleProvider:
    def __init__(self, settings):
        """Сохраняет проверенные настройки адреса, модели, ключа и тайм-аута провайдера."""
        self.settings = settings

    async def generate(self, messages: list[dict], repair: bool = False) -> str:
        """Отправляет сообщения в OpenAI-совместимый chat/completions и возвращает LLMText.

        При repair добавляет системную инструкцию исправить предыдущую генерацию.
        Если провайдер сообщил usage, прикрепляет целочисленные счётчики токенов.
        Пустой, некорректный или слишком длинный ответ и ошибки HTTP превращает
        в ProviderUnavailable без тела ошибки провайдера.
        """
        payload = generation_messages(messages, repair)
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.llm_timeout_seconds, follow_redirects=False
            ) as client:
                body = await json_request(
                    client,
                    "POST",
                    self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                    json={
                        **(getattr(self.settings, "llm_extra_body", None) or {}),
                        "model": self.settings.llm_model,
                        "messages": payload,
                        "temperature": 0.4,
                        "max_tokens": getattr(self.settings, "llm_max_output_tokens", 600),
                    },
                )
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip() or len(text) > 10000:
                    raise ValueError("Invalid output")
                result = LLMText(text)
                usage = body.get("usage")
                if isinstance(usage, dict):
                    result.usage = {
                        key: usage[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0
                    }
                return result
        except (ProviderUnavailable, httpx.HTTPError, KeyError, ValueError, TypeError, IndexError):
            raise ProviderUnavailable("llm_unavailable") from None

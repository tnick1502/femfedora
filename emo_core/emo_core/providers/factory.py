from emo_core.providers.jev_provider import TypeSafeJevProvider
from emo_core.providers.llm_provider import OpenAICompatibleProvider
from emo_core.providers.mock_providers import (
    DisabledJevProvider,
    DisabledLLMProvider,
    MockJevProvider,
    MockLLMProvider,
)


def providers(settings):
    """Создаёт пару (LLM, Jev) для режима mock, live или disabled без сетевых вызовов."""
    if settings.ai_mode == "mock":
        return MockLLMProvider(), MockJevProvider()
    if settings.ai_mode == "live":
        return OpenAICompatibleProvider(settings), TypeSafeJevProvider(settings)
    return DisabledLLMProvider(), DisabledJevProvider()

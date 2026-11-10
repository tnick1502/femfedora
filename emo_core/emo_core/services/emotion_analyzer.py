from emo_core.core.exceptions import ProviderUnavailable
from emo_core.schemas.emotion import EmotionAnalysis


class EmotionAnalyzer:
    def __init__(self, provider):
        """Сохраняет провайдера анализа эмоциональных и защитных маркеров."""
        self.provider = provider

    async def analyze(self, text: str) -> EmotionAnalysis:
        """Запрашивает анализ текста или возвращает available=False при сбое провайдера.

        В ошибочном результате сохраняет только известный безопасный код ошибки;
        оценки маркеров при недоступности не придумывает.
        """
        try:
            return await self.provider.analyze(text)
        except ProviderUnavailable as error:
            code = str(error)
            allowed = {"jev_unavailable", "jev_invalid_response", "external_ai_disabled"}
            return EmotionAnalysis(
                available=False, error_code=code if code in allowed else "classifier_unavailable"
            )

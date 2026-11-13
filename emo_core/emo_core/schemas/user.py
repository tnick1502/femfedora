from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrivacySettings(StrictModel):
    """Обезличивание текста перед отправкой внешним моделям."""

    anonymize: bool = True
    level: Literal["standard", "strict"] = "standard"
    # Слова, которые пользователь просит скрывать всегда: имена, город, место работы.
    protected_terms: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("protected_terms")
    @classmethod
    def clean_terms(cls, value):
        """Нормализует пробелы и удаляет повторы слов для обезличивания без учёта регистра.

        Каждая строка после нормализации должна содержать от 2 до 60 символов;
        неподходящая длина вызывает ValueError.
        """
        cleaned = []
        for term in value:
            term = " ".join(str(term).split())
            if not 2 <= len(term) <= 60:
                raise ValueError("Каждое слово для скрытия — от 2 до 60 символов")
            if term.lower() not in {t.lower() for t in cleaned}:
                cleaned.append(term)
        return cleaned


class ContextSettings(StrictModel):
    """Управление размером контекста, уходящего в LLM."""

    compression: Literal["off", "auto", "compact"] = "auto"
    # 0 — по окну модели (окно − резерв − ответ); меньшее значение сжимает раньше.
    input_target_tokens: int = Field(0, ge=0, le=4_000_000)
    keep_recent_messages: int = Field(12, ge=2, le=200)


class SettingsPatch(StrictModel):
    privacy: PrivacySettings | None = None
    context: ContextSettings | None = None
    timezone: str | None = None
    proactive_enabled: bool | None = None
    push_enabled: bool | None = None
    external_ai_consent: bool | None = None
    quiet_hours_start: time | None = None
    quiet_hours_end: time | None = None
    proactive_paused_until: datetime | None = None
    pregnancy_id: UUID | None = None
    gestational_week: int | None = Field(None, ge=0, le=45)
    estimated_due_date: date | None = None
    birth_date: datetime | None = None
    pregnancy_outcome: Literal["ongoing", "live_birth", "loss"] | None = None
    reproductive_stage: (
        Literal["preconception", "pregnant", "postpartum", "after_loss", "unknown"] | None
    ) = None
    country_code: str | None = Field(None, pattern=r"^[A-Z]{2}$")

    @field_validator("timezone")
    @classmethod
    def timezone_valid(cls, value):
        """Проверяет имя часового пояса через ZoneInfo; None разрешён на уровне схемы."""
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError):
                raise ValueError("Ожидается IANA timezone") from None
        return value

    @field_validator("birth_date", "proactive_paused_until")
    @classmethod
    def aware(cls, value):
        """Отклоняет дату без часового пояса; None и даты с поясом возвращает без изменений."""
        if value is not None and value.tzinfo is None:
            raise ValueError("Укажите часовой пояс")
        return value

    @field_validator("birth_date")
    @classmethod
    def birth_not_in_future(cls, value):
        """Отклоняет дату родов более чем на сутки позже текущего времени UTC."""
        if value is not None and value > datetime.now(UTC) + timedelta(days=1):
            raise ValueError("Дата родов не может быть в будущем; используйте estimated_due_date")
        return value

    @field_validator("estimated_due_date")
    @classmethod
    def due_date_plausible(cls, value):
        """Проверяет предполагаемую дату родов: от 30 дней назад до 310 дней вперёд по UTC."""
        if value is not None:
            today = datetime.now(UTC).date()
            if not today - timedelta(days=30) <= value <= today + timedelta(days=310):
                raise ValueError("Предполагаемая дата родов вне допустимого диапазона")
        return value

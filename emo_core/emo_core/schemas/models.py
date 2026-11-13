from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr, field_validator, model_validator

from emo_core.schemas.user import StrictModel


class ModelCreate(StrictModel):
    kind: Literal["llm", "jev"]
    name: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=160, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./:-]*$")
    base_url: str = Field(min_length=1, max_length=500)
    timeout_seconds: float = Field(default=30, ge=1, le=120)
    credential_ref: str | None = Field(default=None, min_length=1, max_length=100)
    api_key: SecretStr | None = Field(default=None, json_schema_extra={"writeOnly": True})

    @model_validator(mode="after")
    def credential_source(self):
        """Требует ровно один источник ключа: api_key либо credential_ref; возвращает модель."""
        if bool(self.credential_ref) == bool(self.api_key and self.api_key.get_secret_value()):
            raise ValueError("Укажите API-ключ или ссылку на ключ из .env")
        return self

    @field_validator("api_key")
    @classmethod
    def key_valid(cls, value):
        """Проверяет длину ключа 1–4096 символов и отсутствие пробелов; сохраняет SecretStr."""
        if value and (
            not 1 <= len(value.get_secret_value()) <= 4096
            or any(c.isspace() for c in value.get_secret_value())
        ):
            raise ValueError("Некорректный API-ключ")
        return value


class ModelPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    model: str | None = Field(
        default=None, min_length=1, max_length=160, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./:-]*$"
    )
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    timeout_seconds: float | None = Field(default=None, ge=1, le=120)
    credential_ref: str | None = Field(default=None, min_length=1, max_length=100)
    api_key: SecretStr | None = Field(default=None, json_schema_extra={"writeOnly": True})
    _valid_key = field_validator("api_key")(ModelCreate.key_valid.__func__)

    @model_validator(mode="after")
    def non_null_fields(self):
        """Запрещает null в переданных полях и одновременную замену обоих источников ключа."""
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError("Не передавайте null; пропустите поле, чтобы сохранить его")
        if self.credential_ref and self.api_key:
            raise ValueError("Выберите только один источник ключа")
        return self


class RuntimePatch(StrictModel):
    voice_enabled: bool | None = None
    voice_backend: Literal["deepgram", "gigaam"] | None = None
    ai_mode: Literal["mock", "live", "disabled"] | None = None
    llm_profile_id: UUID | None = None
    jev_profile_id: UUID | None = None
    expected_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def non_null_mode(self):
        """Запрещает явный null у ai_mode; пропуск поля оставляет текущий режим."""
        if any(
            key in self.model_fields_set and getattr(self, key) is None
            for key in ("ai_mode", "voice_enabled", "voice_backend")
        ):
            raise ValueError("Укажите значение настройки")
        return self

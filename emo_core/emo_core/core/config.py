from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = Field(default="postgresql+asyncpg://localhost/emo", repr=False)
    db_pool_size: int = Field(default=10, ge=1, le=200)
    db_max_overflow: int = Field(default=20, ge=0, le=400)
    jwt_issuer: str = "https://identity.example.invalid/"
    jwt_audience: str = "emo-core"
    jwt_jwks_url: str = ""
    demo_auth_enabled: bool = False
    demo_jwt_secret: str = Field(default="", repr=False)
    admin_api_token: str = Field(default="", repr=False)
    # Вход в консоль (демо и администрирование) по логину и паролю.
    app_login: str = ""
    app_password: str = Field(default="", repr=False)
    console_session_hours: int = Field(default=8, ge=1, le=72)
    api_key_ttl_days: int = Field(default=30, ge=1, le=365)
    # OpenAI-совместимые параметры провайдера, например отключение режима рассуждений.
    llm_extra_body: dict = Field(default_factory=dict)
    model_secrets_key: str = Field(default="", repr=False)
    model_credentials: dict[str, SecretStr] = Field(default_factory=dict, repr=False)
    ai_mode: Literal["mock", "live", "disabled"] = "disabled"
    external_ai_allowed: bool = False
    llm_base_url: str = ""
    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    llm_timeout_seconds: float = 30
    typesafe_api_key: str = Field(default="", repr=False)
    typesafe_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-1.14.0"
    jev_timeout_seconds: float = 15
    jev_threshold: float = 0.7
    jev_review_threshold: float = 0.35
    # Мысли о самоповреждении и срочные телесные признаки проверяются с более низким порогом:
    # пропуск дороже ложной тревоги. На живом Jev (10.11.2026, 80 реплик) обычные реплики
    # давали меньше 0,08, слабые формулировки («не хочется просыпаться») — от 0,17.
    jev_sensitive_review_threshold: float = 0.12
    # Порог блокировки ответа модели по оценкам Jev (отдельно от порогов входа).
    jev_output_threshold: float = 0.5
    jev_max_retries: int = Field(default=1, ge=0, le=3)
    llm_max_output_tokens: int = Field(default=600, ge=64, le=8000)
    # Окно контекста одного запроса к LLM и резерв; цель входа пользователь задаёт в кабинете.
    context_window_tokens: int = Field(default=256_000, ge=8_192, le=4_000_000)
    context_safety_margin_tokens: int = Field(default=16_000, ge=0)
    # user — решает пользователь (по умолчанию включено); always / never — задаёт оператор.
    anonymize_outbound: Literal["user", "always", "never"] = "user"
    allowed_llm_hosts: str = ""
    allowed_jev_hosts: str = "api.typesafe.ai"
    ai_processing_region: str = ""
    proactive_worker_enabled: bool = True
    proactive_poll_seconds: int = 300
    retention_days: int = 365
    policy_dir: str = ""

    @model_validator(mode="after")
    def validate_runtime(self):
        """Проверяет совместимость режима, авторизации и настроек провайдеров.

        В production запрещены mock и демо-вход; для live обязательны ключи,
        регион обработки и HTTPS-адреса из списка разрешённых хостов.
        Проверяет длину секретов, ключ Fernet и взаимное расположение порогов Jev.
        Возвращает экземпляр настроек; при нарушении условий вызывает ValueError.
        """
        if self.demo_auth_enabled and len(self.demo_jwt_secret) < 32:
            raise ValueError("Set DEMO_JWT_SECRET to at least 32 characters in .env")
        if bool(self.app_login) != bool(self.app_password):
            raise ValueError("Set both APP_LOGIN and APP_PASSWORD or neither")
        if self.app_password and (len(self.app_password) < 12 or len(self.demo_jwt_secret) < 32):
            raise ValueError(
                "APP_PASSWORD needs at least 12 characters and DEMO_JWT_SECRET of 32+ characters"
            )
        if self.admin_api_token and len(self.admin_api_token) < 32:
            raise ValueError("ADMIN_API_TOKEN must contain at least 32 characters")
        if self.model_secrets_key:
            from cryptography.fernet import Fernet

            try:
                Fernet(self.model_secrets_key.encode())
            except (ValueError, TypeError):
                raise ValueError("MODEL_SECRETS_KEY must be a valid Fernet key") from None
        if self.app_env == "production":
            if self.demo_auth_enabled or self.ai_mode == "mock":
                raise ValueError("Production forbids demo authentication and mock AI")
            if not self.jwt_jwks_url.startswith("https://"):
                raise ValueError("Production requires HTTPS JWT_JWKS_URL")
            if not self.database_url.startswith("postgresql+asyncpg://"):
                raise ValueError("Production requires PostgreSQL/asyncpg")
        if not 0 < self.jev_review_threshold < self.jev_threshold <= 1:
            raise ValueError("Invalid Jev thresholds")
        if not 0 < self.jev_sensitive_review_threshold <= self.jev_review_threshold:
            raise ValueError("JEV_SENSITIVE_REVIEW_THRESHOLD must not exceed the review threshold")
        if not 0 < self.jev_output_threshold <= 1:
            raise ValueError("Invalid JEV_OUTPUT_THRESHOLD")
        if self.proactive_poll_seconds < 1 or self.retention_days < 1:
            raise ValueError("Polling/retention must be positive")
        if self.jev_model == "jev-latest":
            raise ValueError("Pin a reviewed JEV_MODEL version")
        if self.ai_mode == "live":
            if not all(
                (
                    self.external_ai_allowed,
                    self.llm_api_key,
                    self.llm_model,
                    self.typesafe_api_key,
                    self.ai_processing_region,
                )
            ):
                raise ValueError(
                    "Live AI requires explicit consent, keys, model and processing region"
                )
            for url, hosts in (
                (self.llm_base_url, self.allowed_llm_hosts),
                (self.typesafe_base_url, self.allowed_jev_hosts),
            ):
                parsed = urlparse(url)
                if parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError(
                        "Do not put credentials or query parameters in AI endpoint URLs"
                    )
                if parsed.scheme != "https" or parsed.hostname not in {
                    h.strip().lower() for h in hosts.split(",")
                }:
                    raise ValueError("AI endpoint must use HTTPS and an explicitly allowed host")
        return self

    @property
    def login_required(self) -> bool:
        """Возвращает True, если заданы оба поля входа в консоль: логин и пароль."""
        return bool(self.app_login and self.app_password)


@lru_cache
def get_settings() -> Settings:
    """Загружает настройки из окружения и .env, проверяет их и кеширует в процессе."""
    return Settings()

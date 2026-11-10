from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import aliased

from emo_core.core.config import Settings
from emo_core.core.secrets import SecretConfigurationError, decrypt_key, encrypt_key, reveal
from emo_core.db.models import ModelConfigEvent, ModelProfile, ModelRuntime


@dataclass
class RuntimeSnapshot:
    settings: Settings
    models: dict
    revision: int
    overridden: bool


class ModelRegistry:
    def __init__(self, settings: Settings):
        """Сохраняет базовые настройки окружения для разрешения общих профилей моделей."""
        self.settings = settings

    async def control(self, session, lock=False):
        """Возвращает единственную строку общей runtime-конфигурации, создавая её при отсутствии.

        При lock=True обновляет ORM-состояние из БД и берёт FOR UPDATE до конца
        транзакции; этот режим нужен при административных изменениях.
        """
        row = await session.get(ModelRuntime, 1)
        if row is None:
            insert = pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert
            await session.execute(
                insert(ModelRuntime).values(id=1, revision=0).on_conflict_do_nothing()
            )
        query = select(ModelRuntime).where(ModelRuntime.id == 1)
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        return await session.scalar(query)

    def environment_keys(self):
        """Собирает именованные секреты LLM/Jev и дополнительные model_credentials.

        Возвращает открытые значения для внутреннего использования; словарь
        нельзя сериализовать в API-ответ или журнал.
        """
        return {
            "llm_default": self.settings.llm_api_key,
            "jev_default": self.settings.typesafe_api_key,
            **{
                key: reveal(value)
                for key, value in self.settings.model_credentials.items()
                if key not in ("llm_default", "jev_default")
            },
        }

    def key(self, profile):
        """Читает секрет профиля из env-ссылки либо расшифровывает ключ из БД.

        Неизвестная или пустая ссылка, отсутствие ключа и ошибка расшифровки
        вызывают SecretConfigurationError.
        """
        if profile.credential_ref:
            key = self.environment_keys().get(profile.credential_ref, "")
            if not key:
                raise SecretConfigurationError("У выбранной ссылки нет ключа в .env")
            return key
        if not profile.encrypted_api_key:
            raise SecretConfigurationError("API-ключ не задан")
        return decrypt_key(profile.encrypted_api_key, self.settings.model_secrets_key)

    def validate_profile(self, profile):
        """Проверяет HTTPS-адрес, разрешённый хост, закреплённую версию Jev и доступность ключа.

        Запрещает учётные данные, query и fragment в URL. Ошибки конфигурации
        возвращает как HTTP 422; сетевого запроса к модели не выполняет.
        """
        parsed = urlparse(profile.base_url)
        hosts = (
            self.settings.allowed_llm_hosts
            if profile.kind == "llm"
            else self.settings.allowed_jev_hosts
        )
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {h.strip().lower() for h in hosts.split(",")}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise HTTPException(
                422,
                "Endpoint должен использовать HTTPS и host из ALLOWED_LLM_HOSTS/ALLOWED_JEV_HOSTS; секреты в URL запрещены",
            )
        if profile.kind == "jev" and profile.model == "jev-latest":
            raise HTTPException(422, "Для Jev укажите закреплённую версию")
        try:
            self.key(profile)
        except SecretConfigurationError as error:
            raise HTTPException(422, str(error)) from None

    async def resolve(self, session) -> RuntimeSnapshot:
        """Собирает согласованный снимок моделей и настроек из БД и окружения.

        Выбранные профили и режим перекрывают .env. Одним запросом читает ревизию
        и оба профиля; секреты расшифровывает только для live. Возвращает
        RuntimeSnapshot с внутренними настройками и публичными описаниями моделей.
        Недоступный ключ даёт HTTP 503, несовместимая конфигурация — 409.
        """
        await self.control(session)
        llm, jev = aliased(ModelProfile), aliased(ModelProfile)
        # One MVCC statement gives a consistent revision/profile snapshot across API replicas.
        row, llm_profile, jev_profile = (
            await session.execute(
                select(ModelRuntime, llm, jev)
                .outerjoin(llm, ModelRuntime.llm_profile_id == llm.id)
                .outerjoin(jev, ModelRuntime.jev_profile_id == jev.id)
                .where(ModelRuntime.id == 1)
                .execution_options(populate_existing=True)
            )
        ).one()
        values = self.settings.model_dump()
        values["ai_mode"] = row.ai_mode or self.settings.ai_mode
        for profile in (llm_profile, jev_profile):
            if profile is None:
                continue
            api_key = ""
            if values["ai_mode"] == "live":
                try:
                    api_key = self.key(profile)
                except SecretConfigurationError:
                    raise HTTPException(
                        503,
                        "Ключ активной модели недоступен; администратору нужно проверить конфигурацию",
                    ) from None
            prefix = "llm" if profile.kind == "llm" else "jev"
            values[f"{prefix}_model"] = profile.model
            values[f"{prefix}_timeout_seconds"] = profile.timeout_seconds
            values["llm_base_url" if profile.kind == "llm" else "typesafe_base_url"] = (
                profile.base_url
            )
            values["llm_api_key" if profile.kind == "llm" else "typesafe_api_key"] = api_key
        try:
            effective = Settings.model_validate(values)
        except ValidationError:
            raise HTTPException(
                409,
                "Режим не может быть применён: проверьте ключи обеих моделей, EXTERNAL_AI_ALLOWED, allowlist и AI_PROCESSING_REGION в .env; production запрещает mock",
            ) from None
        mode = effective.ai_mode
        descriptions = {}
        for kind, profile in (("llm", llm_profile), ("jev", jev_profile)):
            configured_model = effective.llm_model if kind == "llm" else effective.jev_model
            actual = (
                configured_model
                if mode == "live"
                else ("deterministic-support-v1" if kind == "llm" else "deterministic-demo-v1")
                if mode == "mock"
                else None
            )
            descriptions[kind] = {
                "provider": ("openai_compatible" if kind == "llm" else "typesafe")
                if mode == "live"
                else mode,
                "model": actual,
                "configured_model": configured_model or None,
                "profile_id": str(profile.id) if profile else None,
                "profile_name": profile.name if profile else "Настройки .env",
                "source": "registry" if profile else "env",
            }
        return RuntimeSnapshot(
            effective,
            {"ai_mode": mode, "revision": row.revision, **descriptions},
            row.revision,
            bool(row.ai_mode or llm_profile or jev_profile),
        )

    @staticmethod
    def describe(profile, control):
        """Возвращает поля профиля, источник ключа и признак активности без секрета и шифротекста."""
        return {
            "id": profile.id,
            "kind": profile.kind,
            "name": profile.name,
            "model": profile.model,
            "base_url": profile.base_url,
            "timeout_seconds": profile.timeout_seconds,
            "credential_ref": profile.credential_ref,
            "has_api_key": bool(profile.credential_ref or profile.encrypted_api_key),
            "key_source": "env" if profile.credential_ref else "encrypted",
            "active": getattr(control, f"{profile.kind}_profile_id") == profile.id,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        }

    async def event(self, session, control, action, profile=None, fields=None):
        """Повышает ревизию и сохраняет действие и имена изменённых полей без их значений."""
        control.revision += 1
        session.add(
            ModelConfigEvent(
                revision=control.revision,
                action=action,
                profile_id=profile.id if profile else None,
                details={"fields": sorted(fields or []), "kind": profile.kind if profile else None},
            )
        )
        await session.flush()

    def apply_secret(self, profile, values):
        """Заменяет источник секрета профиля только при переданном непустом значении.

        Env-ссылка удаляет прежний шифротекст; новый api_key шифруется Fernet
        и удаляет ссылку. Пропущенный ключ сохраняется. Ошибка шифрования даёт 422.
        """
        if values.get("credential_ref"):
            profile.credential_ref = values["credential_ref"]
            profile.encrypted_api_key = None
        if values.get("api_key"):
            try:
                profile.encrypted_api_key = encrypt_key(
                    reveal(values["api_key"]), self.settings.model_secrets_key
                )
            except SecretConfigurationError as error:
                raise HTTPException(422, str(error)) from None
            profile.credential_ref = None

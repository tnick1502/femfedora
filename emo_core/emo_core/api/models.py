from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.admin import admin_session
from emo_core.db.models import ModelConfigEvent, ModelProfile
from emo_core.db.repositories import public, tenant_session
from emo_core.schemas.models import ModelCreate, ModelPatch, RuntimePatch
from emo_core.services.model_registry import ModelRegistry

router = APIRouter()


@router.get("/models")
async def models(
    request: Request, session: AsyncSession = Depends(tenant_session, scope="function")
):
    """Возвращает действующие модели, режим AI и ревизию без значений API-ключей."""
    return (await ModelRegistry(request.app.state.settings).resolve(session)).models


@router.get("/admin/models")
async def list_models(
    request: Request, session: AsyncSession = Depends(admin_session, scope="function")
):
    """Возвращает администратору профили моделей и состояние общей конфигурации.

    Вместо ключей отдаёт имена env-ссылок и признаки наличия секретов.
    Ошибка текущей конфигурации возвращается в runtime, чтобы её можно было
    исправить через панель.
    """
    registry = ModelRegistry(request.app.state.settings)
    control = await registry.control(session)
    profiles = list(
        (await session.scalars(select(ModelProfile).order_by(ModelProfile.created_at))).all()
    )
    try:
        runtime = (await registry.resolve(session)).models
    except HTTPException as error:
        runtime = {
            "ai_mode": control.ai_mode or request.app.state.settings.ai_mode,
            "revision": control.revision,
            "configuration_error": error.detail,
        }
    return {
        "items": [registry.describe(p, control) for p in profiles],
        "runtime": runtime,
        "credential_refs": [
            {"name": key, "configured": bool(value)}
            for key, value in registry.environment_keys().items()
        ],
        "key_encryption_configured": bool(request.app.state.settings.model_secrets_key),
        "allowed_hosts": {
            "llm": request.app.state.settings.allowed_llm_hosts,
            "jev": request.app.state.settings.allowed_jev_hosts,
        },
    }


@router.post("/admin/models", status_code=201)
async def add_model(
    body: ModelCreate,
    request: Request,
    session: AsyncSession = Depends(admin_session, scope="function"),
):
    """Создаёт профиль провайдера после проверки адреса и источника ключа.

    Ключ сохраняется как env-ссылка либо шифротекст Fernet. Создание повышает
    ревизию реестра и записывается в аудит; активную модель само по себе не меняет.
    """
    registry = ModelRegistry(request.app.state.settings)
    control = await registry.control(session, lock=True)
    profile = ModelProfile(**body.model_dump(exclude={"api_key", "credential_ref"}))
    registry.apply_secret(profile, body.model_dump())
    registry.validate_profile(profile)
    session.add(profile)
    await session.flush()
    await registry.event(session, control, "profile_created", profile, body.model_fields_set)
    return registry.describe(profile, control)


@router.patch("/admin/models/{profile_id}")
async def edit_model(
    profile_id: UUID,
    body: ModelPatch,
    request: Request,
    session: AsyncSession = Depends(admin_session, scope="function"),
):
    """Изменяет существующий профиль модели и записывает поля изменения в аудит.

    Пропущенные поля, включая ключ, сохраняет. Проверяет профиль и итоговую
    runtime-конфигурацию в той же транзакции; отсутствие профиля даёт 404.
    """
    registry = ModelRegistry(request.app.state.settings)
    control = await registry.control(session, lock=True)
    profile = await session.get(ModelProfile, profile_id)
    if profile is None:
        raise HTTPException(404, "Модель не найдена")
    for key, value in body.model_dump(
        exclude_unset=True, exclude={"api_key", "credential_ref"}
    ).items():
        setattr(profile, key, value)
    registry.apply_secret(profile, body.model_dump(exclude_unset=True))
    registry.validate_profile(profile)
    await registry.event(session, control, "profile_updated", profile, body.model_fields_set)
    await registry.resolve(session)
    return registry.describe(profile, control)


@router.patch("/admin/model-runtime")
async def activate(
    body: RuntimePatch,
    request: Request,
    session: AsyncSession = Depends(admin_session, scope="function"),
):
    """Меняет режим и выбранные профили моделей для всех пользователей.

    expected_revision должна совпадать с текущей, иначе возвращается HTTP 409.
    Null у идентификатора профиля возвращает соответствующую модель к .env.
    Проверяет типы профилей и совместимость всей конфигурации перед фиксацией.
    """
    registry = ModelRegistry(request.app.state.settings)
    control = await registry.control(session, lock=True)
    if control.revision != body.expected_revision:
        raise HTTPException(409, "Настройки уже изменились; обновите список моделей")
    for kind in ("llm", "jev"):
        key = f"{kind}_profile_id"
        if key not in body.model_fields_set:
            continue
        profile_id = getattr(body, key)
        if profile_id:
            profile = await session.get(ModelProfile, profile_id)
            if profile is None or profile.kind != kind:
                raise HTTPException(422, "Профиль отсутствует или имеет другой тип")
            registry.validate_profile(profile)
        setattr(control, key, profile_id)
    if body.ai_mode is not None:
        control.ai_mode = body.ai_mode
    await registry.event(
        session, control, "runtime_updated", fields=body.model_fields_set - {"expected_revision"}
    )
    return (await registry.resolve(session)).models


@router.get("/admin/model-events")
async def model_events(
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(admin_session, scope="function"),
):
    """Возвращает администратору страницу аудита моделей по убыванию ревизии."""
    rows = await session.scalars(
        select(ModelConfigEvent)
        .order_by(ModelConfigEvent.revision.desc())
        .limit(limit)
        .offset(offset)
    )
    return {"items": [public(row) for row in rows], "limit": limit, "offset": offset}

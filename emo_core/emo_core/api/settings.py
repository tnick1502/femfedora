from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db import models as m
from emo_core.db.base import utcnow
from emo_core.db.repositories import public, tenant_read_session, tenant_session
from emo_core.schemas.chat import FeedbackCreate, MemoryPatch
from emo_core.schemas.user import (
    ContextSettings,
    PrivacySettings,
    SettingsPatch,
    StrictModel,
    VoiceSettings,
)
from emo_core.services.anonymizer import boundary_for
from emo_core.services.context_manager import COMPRESSION_MODES, ContextBudget
from emo_core.services.memory_manager import MemoryManager
from emo_core.services.patient_state import PatientStateBuilder
from emo_core.services.safety_guard import SafetyGuard

router = APIRouter()


def settings_view(obj, server):
    """Собирает настройки профиля с конфиденциальностью и бюджетом контекста.

    Подставляет значения по умолчанию в отсутствующие разделы ai_preferences
    и добавляет серверные ограничения. ORM-объект при этом не изменяет.
    """
    data = public(obj)
    stored = data.pop("ai_preferences") or {}
    privacy = PrivacySettings(**(stored.get("privacy") or {})).model_dump()
    context = ContextSettings(**(stored.get("context") or {})).model_dump()
    budget = ContextBudget.from_settings(
        server, {"context": context}, max_output_tokens=server.llm_max_output_tokens
    )
    return {
        **data,
        "privacy": privacy,
        "voice": VoiceSettings(**(stored.get("voice") or {})).model_dump(),
        # Имена, которые система сама встретила в репликах и скрывает впредь.
        "learned_names": stored.get("learned_names") or [],
        "context": context,
        "limits": {
            **budget.as_dict(),
            "compression_modes": COMPRESSION_MODES,
            # always / never — обезличиванием управляет оператор, переключатель не действует.
            "anonymize_policy": server.anonymize_outbound,
        },
    }


class PrivacyPreview(StrictModel):
    text: str = Field(min_length=1, max_length=4000)


@router.get("/settings")
async def settings(
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает настройки владельца JWT вместе с действующими серверными пределами."""
    return settings_view(await session.get(m.UserSettings, user_id), request.app.state.settings)


@router.get("/support-contacts")
async def support_contacts(
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает контакты из safety policy для country_code владельца JWT."""
    obj = await session.get(m.UserSettings, user_id)
    return SafetyGuard(request.app.state.settings, None).contacts(obj.country_code)


@router.post("/privacy/preview")
async def privacy_preview(
    body: PrivacyPreview,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Показывает замену персональных данных по текущим правилам профиля.

    Возвращает обезличенный текст и количество замен по типам.
    Текст не сохраняется и не отправляется внешним моделям.
    """
    server = request.app.state.settings
    user = await session.get(m.UserSettings, user_id)
    phones, _ = SafetyGuard(server, None).allowed_contacts()
    return boundary_for(server, user.ai_preferences, phones).preview(body.text)


@router.delete("/privacy/learned-names")
async def forget_learned_names(
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Удаляет запомненные имена из ai_preferences, сохраняя остальные настройки."""
    obj = await session.get(m.UserSettings, user_id)
    obj.ai_preferences = {
        k: v for k, v in (obj.ai_preferences or {}).items() if k != "learned_names"
    }
    return {"status": "deleted"}


@router.patch("/settings")
async def patch(
    body: SettingsPatch,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Применяет переданные настройки владельца JWT; пропущенные поля сохраняет.

    Проверяет цель контекста по серверному бюджету, привязывает срок беременности
    к дате передачи и отмечает пользовательскую паузу отдельной причиной.
    Выключение инициативных сообщений отменяет ожидающие задания.
    Недопустимый null или превышение бюджета возвращает HTTP 422.
    """
    obj = await session.get(m.UserSettings, user_id)
    nullable = {
        "proactive_paused_until",
        "pregnancy_id",
        "gestational_week",
        "estimated_due_date",
        "birth_date",
        "pregnancy_outcome",
        "country_code",
        "reproductive_stage",
    }
    changes = body.model_dump(exclude_unset=True)
    # Resolve inferred stage BEFORE clearing incompatible calendar data.
    new_episode = "pregnancy_id" in changes and changes["pregnancy_id"] != obj.pregnancy_id
    if new_episode:
        for key in (
            "gestational_week",
            "estimated_due_date",
            "birth_date",
            "pregnancy_outcome",
            "reproductive_stage",
        ):
            changes.setdefault(key, None)
    if (
        "reproductive_stage" not in body.model_fields_set
        or changes.get("reproductive_stage") is None
    ):
        if changes.get("pregnancy_outcome") == "loss":
            changes["reproductive_stage"] = "after_loss"
        elif changes.get("birth_date"):
            changes["reproductive_stage"] = "postpartum"
        elif changes.get("estimated_due_date") or changes.get("gestational_week") is not None:
            changes["reproductive_stage"] = "pregnant"
    stage = changes.get("reproductive_stage")
    if stage in ("preconception", "unknown", "after_loss"):
        changes.update(gestational_week=None, estimated_due_date=None, birth_date=None)
        changes["pregnancy_outcome"] = "loss" if stage == "after_loss" else None
    elif stage == "pregnant":
        changes.update(birth_date=None, pregnancy_outcome="ongoing")
    elif stage == "postpartum":
        changes.update(
            gestational_week=None, estimated_due_date=None, pregnancy_outcome="live_birth"
        )
    server = request.app.state.settings
    preferences = dict(obj.ai_preferences or {})
    for section in ("privacy", "context", "voice"):
        if section in changes:
            value = changes.pop(section)
            if value is None:
                preferences.pop(section, None)
                continue
            if section == "context":
                hard = ContextBudget.from_settings(
                    server, None, max_output_tokens=server.llm_max_output_tokens
                ).hard_input_budget
                if value["input_target_tokens"] > hard:
                    raise HTTPException(
                        422,
                        f"Цель входа не может превышать {hard} токенов для текущего окна модели",
                    )
            preferences[section] = value
    obj.ai_preferences = preferences
    for key, value in changes.items():
        if value is None and key not in nullable:
            raise HTTPException(422, f"{key} не может быть null")
        setattr(obj, key, value)
    if "gestational_week" in changes:
        # Срок привязывается к моменту передачи, иначе он перестаёт соответствовать календарю.
        obj.gestational_week_recorded_at = (
            utcnow() if changes["gestational_week"] is not None else None
        )
    if "proactive_paused_until" in changes:
        # Пауза, заданная пользователем, не снимается автоматически при новом сообщении.
        obj.proactive_pause_reason = "user" if changes["proactive_paused_until"] else None
    if (
        changes.get("pregnancy_id")
        and obj.pregnancy_outcome == "loss"
        and "pregnancy_outcome" not in changes
    ):
        # Новая беременность после потери: прежний исход к ней не относится.
        obj.pregnancy_outcome = None
    if not obj.proactive_enabled:
        jobs = await session.scalars(
            select(m.ProactiveJob).where(
                m.ProactiveJob.user_id == user_id, m.ProactiveJob.status == "pending"
            )
        )
        for job in jobs:
            job.status, job.last_error = "cancelled", "consent_disabled"
    await session.flush()
    return settings_view(obj, server)


@router.get("/state")
async def state(
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Возвращает или создаёт память профиля, отмечая данные как недиагностические."""
    return {
        **public(await MemoryManager().get(session, user_id)),
        "patient_context": await PatientStateBuilder(request.app.state.settings).build(
            session, user_id
        ),
        "not_a_diagnosis": True,
    }


@router.patch("/memory")
async def memory(
    body: MemoryPatch,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Заменяет резюме и предпочтения профиля пользовательской редакцией.

    Очищает извлечённые факты и повторяющиеся темы, обновляет временную границу
    памяти, чтобы старые сообщения не восстановили удалённые сведения.
    """
    obj = await MemoryManager().get(session, user_id)
    obj.summary, obj.preferences = body.summary, body.preferences
    obj.updated_at = obj.memory_cleared_at = utcnow()
    obj.facts = (
        [{"source": "user_edit", "text": body.summary, "observed_at": obj.updated_at.isoformat()}]
        if body.summary
        else []
    )
    obj.recurring_topics = []
    return public(obj)


@router.delete("/memory")
async def clear_memory(
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Очищает память профиля, сохраняя переписку.

    Запоминает время очистки и идентификатор последней реплики: последующее
    обновление памяти не должно заново разобрать историю до этой границы.
    """
    obj = await MemoryManager().get(session, user_id)
    obj.summary, obj.preferences, obj.facts, obj.recurring_topics = "", {}, [], []
    obj.memory_cleared_at = utcnow()
    # Cursor prevents immediate rehydration from the first historical message.
    obj.last_summarized_message_id = await session.scalar(
        select(m.Message.id)
        .where(m.Message.user_id == user_id)
        .order_by(m.Message.created_at.desc())
        .limit(1)
    )
    return {"status": "deleted"}


@router.post("/feedback", status_code=201)
async def feedback(
    body: FeedbackCreate,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Сохраняет субъективную оценку собственного ответа ассистента.

    Не принимает оценку чужого или пользовательского сообщения; в этих случаях
    возвращает HTTP 404. Оценка не считается результатом клинического измерения.
    """
    message = await session.scalar(
        select(m.Message).where(
            m.Message.user_id == user_id,
            m.Message.id == body.message_id,
            m.Message.role == "assistant",
        )
    )
    if message is None:
        raise HTTPException(404, "Сообщение не найдено")
    row = m.SupportFeedback(user_id=user_id, **body.model_dump())
    session.add(row)
    await session.flush()
    return public(row)


@router.delete("/data")
async def remove_data(
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    # Delete dependents first; retain only a tombstone so a still-valid JWT cannot reactivate the profile.
    """Удаляет данные EMO владельца JWT и деактивирует профиль.

    Связанные записи удаляются в порядке зависимостей; в UserSettings остаётся
    метка деактивации, чтобы действующий JWT не создал профиль заново.
    Резервные копии этой операцией не удаляются.
    """
    for model in (
        m.NotificationOutbox,
        m.SupportFeedback,
        m.PromptTrace,
        m.SafetyEvent,
        m.ProactiveJob,
        m.MessageSignal,
        m.Complaint,
        m.Checkin,
        m.TestAttempt,
        m.AssessmentSchedule,
        m.UserState,
        m.VoiceInput,
        m.Message,
        m.Conversation,
    ):
        await session.execute(delete(model).where(model.user_id == user_id))
    user = await session.get(m.UserSettings, user_id)
    user.proactive_enabled = user.push_enabled = user.external_ai_consent = False
    user.deleted_at = utcnow()
    user.pregnancy_id = user.gestational_week = user.birth_date = user.country_code = None
    user.gestational_week_recorded_at = user.estimated_due_date = user.pregnancy_outcome = None
    user.reproductive_stage = None
    user.timezone, user.proactive_paused_until, user.proactive_pause_reason = "UTC", None, None
    user.ai_preferences = {}
    return {
        "status": "deleted",
        "profile_deactivated": True,
        "note": "Удалены данные EMO; остаётся идентификатор деактивации. Резервные копии удаляются по политике оператора.",
    }

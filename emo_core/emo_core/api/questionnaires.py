from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.base import utcnow
from emo_core.db.models import Questionnaire, TestAttempt, UserSettings
from emo_core.db.repositories import tenant_read_session, tenant_session
from emo_core.schemas.screening import (
    AttemptCreate,
    AttemptPage,
    AttemptUpdate,
    AttemptView,
    QuestionnaireCatalog,
    QuestionnaireView,
    ScheduleAction,
)
from emo_core.services.perinatal import perinatal_context
from emo_core.services.questionnaire_materials import questionnaire_view
from emo_core.services.screening_engine import ScreeningEngine
from emo_core.services.screening_scheduler import (
    SCALE_CODES,
    STAGE_SCALES,
    ScreeningScheduler,
)

router = APIRouter(tags=["Questionnaires"])


@router.get("/questionnaires", response_model=QuestionnaireCatalog)
async def questionnaires(
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает доступные редакции шкал и причины недоступности остальных кодов."""
    rows = list(
        (
            await session.scalars(
                select(Questionnaire).order_by(Questionnaire.code, Questionnaire.version)
            )
        ).all()
    )
    stage = perinatal_context(await session.get(UserSettings, user_id), utcnow()).stage
    allowed = STAGE_SCALES[stage]
    return {
        "items": [questionnaire_view(q) for q in rows if q.enabled and q.code in allowed],
        "unavailable": [
            {
                "code": q.code,
                "version": q.version,
                "reason": q.definition.get("blocked_reason", "Отключён администратором"),
            }
            for q in rows
            if q.code in allowed
            and not q.enabled
            and q.code not in {row.code for row in rows if row.enabled}
        ],
    }


@router.get("/questionnaires/{questionnaire_id}", response_model=QuestionnaireView)
async def questionnaire_detail(
    questionnaire_id: UUID,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает конкретную редакцию шкалы, включая отключённую; отсутствие даёт 404."""
    questionnaire = await session.get(Questionnaire, questionnaire_id)
    if questionnaire is None:
        raise HTTPException(404, "Опросник не найден")
    return questionnaire_view(questionnaire)


@router.post("/questionnaire-attempts", response_model=AttemptView)
async def save_attempt(
    body: AttemptCreate,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Начинает или сохраняет прохождение через совместимый общий endpoint.

    questionnaire_id выбирает новую попытку, attempt_id — существующую.
    client_request_id защищает начало от повторов; complete запрашивает
    завершение. Подсчёт и проверки выполняет ScreeningEngine.
    """
    return await ScreeningEngine(request.app.state.settings, request.app.state.jev).save(
        session, user_id, body
    )


@router.get("/questionnaire-attempts/{attempt_id}", response_model=AttemptView)
async def attempt_detail(
    attempt_id: UUID,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает собственную попытку со снимком формы, ответами, прогрессом и итогом."""
    return await ScreeningEngine(request.app.state.settings, request.app.state.jev).detail(
        session, user_id, attempt_id
    )


@router.patch("/questionnaire-attempts/{attempt_id}", response_model=AttemptView)
async def update_attempt(
    attempt_id: UUID,
    body: AttemptUpdate,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Сохраняет часть ответов собственного черновика с проверкой ревизии.

    Переданные ключи объединяются с сохранёнными. Устаревшая ревизия или
    завершённая попытка даёт HTTP 409; чужая и отсутствующая — 404.
    """
    return await ScreeningEngine(request.app.state.settings, request.app.state.jev).save(
        session, user_id, AttemptCreate(attempt_id=attempt_id, **body.model_dump())
    )


@router.post("/questionnaire-attempts/{attempt_id}/complete", response_model=AttemptView)
async def complete_attempt(
    attempt_id: UUID,
    body: AttemptUpdate,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Завершает собственную попытку и возвращает фиксированный отчёт подсчёта.

    Требует всех обязательных ответов. Повтор без изменения ответов возвращает
    исходный результат; повтор с изменениями завершённой попытки даёт 409.
    """
    return await ScreeningEngine(request.app.state.settings, request.app.state.jev).save(
        session,
        user_id,
        AttemptCreate(attempt_id=attempt_id, complete=True, **body.model_dump()),
        idempotent_completion=True,
    )


@router.get("/questionnaire-attempts", response_model=AttemptPage)
async def attempts(
    request: Request,
    status: str | None = Query(None, pattern="^(in_progress|completed)$"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает страницу собственных прохождений с необязательным фильтром status."""
    rows = (
        await session.execute(
            select(TestAttempt, Questionnaire)
            .join(Questionnaire)
            .where(
                TestAttempt.user_id == user_id,
                *([TestAttempt.status == status] if status else []),
            )
            .order_by(TestAttempt.started_at.desc(), TestAttempt.id)
            .offset(offset)
            .limit(limit)
        )
    ).all()
    engine = ScreeningEngine(request.app.state.settings, request.app.state.jev)
    country = await engine.country(session, user_id)
    return {
        "items": [engine.view(a, q, country) for a, q in rows],
        "limit": limit,
        "offset": offset,
    }


@router.get("/next-actions")
async def next_actions(
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает следующие добровольные действия по текущему профилю и расписанию."""
    return {
        "items": await ScreeningScheduler(request.app.state.settings).next_actions(session, user_id)
    }


@router.post("/assessment-schedule/{code}")
async def schedule_action(
    code: str,
    body: ScheduleAction,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Фиксирует предложение, отказ или перенос известной шкалы для владельца JWT."""
    if code not in SCALE_CODES:
        raise HTTPException(404, "Шкала не найдена")
    await ScreeningScheduler(request.app.state.settings).record(session, user_id, code, body.action)
    return {"status": "saved"}

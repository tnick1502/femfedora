from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.base import utcnow
from emo_core.db.models import UserSettings
from emo_core.db.repositories import owned_conversation, tenant_session
from emo_core.schemas.proactive import FollowupCreate
from emo_core.services.proactive_engine import ProactiveMessagingEngine

router = APIRouter()


@router.post("/proactive/followup", status_code=201)
async def followup(
    body: FollowupCreate,
    request: Request,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Планирует согласованный контакт в собственной беседе на будущее время.

    Требует consent в запросе и включённых инициативных сообщений в профиле.
    Ключ из user_id и client_request_id защищает от повторной постановки задания.
    Нет согласия — HTTP 409; время не в будущем — 422.
    """
    await owned_conversation(session, user_id, body.conversation_id)
    settings = await session.get(UserSettings, user_id)
    if not body.consent or not settings.proactive_enabled:
        raise HTTPException(409, "Нужно явное согласие и включённая инициативная коммуникация")
    if body.scheduled_for <= utcnow():
        raise HTTPException(422, "Выберите будущее время")
    engine = ProactiveMessagingEngine(request.app.state.settings, request.app.state.sessions)
    key = f"followup:{user_id}:{body.client_request_id}"
    await engine.enqueue(
        session,
        user_id=user_id,
        conversation_id=body.conversation_id,
        job_type="agreed_followup",
        deduplication_key=key,
        scheduled_for=body.scheduled_for,
        details={"explicit_consent": True},
    )
    return {"status": "scheduled", "deduplication_key": key}

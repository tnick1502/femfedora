from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.base import utcnow
from emo_core.db.models import Complaint, Message, MessageSignal
from emo_core.db.repositories import public, tenant_read_session, tenant_session
from emo_core.schemas.user import StrictModel

router = APIRouter()


class ComplaintUpdate(StrictModel):
    status: Literal["active", "improving", "resolved", "unknown"]
    intensity: float | None = Field(None, ge=0, le=10)
    message_id: UUID


@router.get("/complaints")
async def complaints(
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает жалобы владельца JWT по убыванию времени последнего наблюдения."""
    rows = await session.scalars(
        select(Complaint)
        .where(Complaint.user_id == user_id)
        .order_by(Complaint.last_seen_at.desc())
    )
    return {"items": [public(r) for r in rows]}


@router.patch("/complaints/{complaint_id}")
async def update(
    complaint_id: UUID,
    body: ComplaintUpdate,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Сохраняет подтверждённый пользователем статус жалобы и её интенсивность.

    Изменение связывается с body.message_id — собственной пользовательской
    репликой. Записывает MessageSignal с источником user_report. Для чужой
    или отсутствующей жалобы либо реплики возвращает HTTP 404.
    """
    complaint = await session.scalar(
        select(Complaint).where(Complaint.user_id == user_id, Complaint.id == complaint_id)
    )
    message = await session.scalar(
        select(Message).where(
            Message.user_id == user_id, Message.id == body.message_id, Message.role == "user"
        )
    )
    if not complaint or not message:
        raise HTTPException(404, "Запись не найдена")
    complaint.status, complaint.user_confirmed = body.status, True
    complaint.last_seen_at, complaint.last_user_reported_intensity = utcnow(), body.intensity
    session.add(
        MessageSignal(
            user_id=user_id,
            message_id=message.id,
            signal_code=complaint.category,
            source="user_report",
            confidence=None,
            details={"status": body.status, "intensity": body.intensity},
        )
    )
    return public(complaint)

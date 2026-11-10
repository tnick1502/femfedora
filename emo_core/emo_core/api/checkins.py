from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.models import Checkin
from emo_core.db.repositories import public, tenant_read_session, tenant_session
from emo_core.schemas.screening import CheckinCreate
from emo_core.services.wellbeing_monitor import WellbeingMonitor

router = APIRouter()


@router.post("/checkins", status_code=201)
async def create(
    body: CheckinCreate,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Сохраняет оценки настроения, тревоги и сна владельца JWT по шкале 0–10."""
    return public(await WellbeingMonitor().save(session, user_id, body))


@router.get("/checkins")
async def history(
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает страницу самооценок владельца JWT, начиная с самых новых."""
    rows = await session.scalars(
        select(Checkin)
        .where(Checkin.user_id == user_id)
        .order_by(Checkin.created_at.desc(), Checkin.id)
        .offset(offset)
        .limit(limit)
    )
    return {"items": [public(r) for r in rows], "limit": limit, "offset": offset}

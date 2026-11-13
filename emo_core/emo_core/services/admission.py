"""Bound request memory and billable work without additional infrastructure."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import func, or_, select

from emo_core.db.base import utcnow
from emo_core.db.models import Message


@asynccontextmanager
async def capacity(slots: asyncio.Semaphore, detail: str):
    if slots.locked():
        raise HTTPException(429, detail, headers={"Retry-After": "5"})
    await slots.acquire()
    try:
        yield
    finally:
        slots.release()


async def chat_budget(session, user_id, settings, request_id):
    """Call under the user lock. Completed replays bypass admission in the caller."""
    cutoff = utcnow() - timedelta(hours=1)
    used = await session.scalar(
        select(func.count())
        .select_from(Message)
        .where(
            Message.user_id == user_id,
            Message.role == "user",
            or_(Message.client_message_id.is_(None), Message.client_message_id != request_id),
            or_(
                Message.created_at >= cutoff,
                Message.details["processing_started_at"].as_string() >= cutoff.isoformat(),
            ),
        )
    )
    if used >= settings.chat_messages_per_hour:
        raise HTTPException(
            429, "Лимит сообщений за час исчерпан. Повторите позже.", headers={"Retry-After": "60"}
        )

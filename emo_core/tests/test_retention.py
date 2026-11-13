from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin
from emo_core.services.retention import prune
from tests.conftest import conversation, send


async def test_pruning_sources_clears_derived_summaries_and_newer_traces(client, auth, sessions):
    from emo_core.db.models import Conversation, Message, PromptTrace

    cid = await conversation(client, auth)
    await send(client, auth, cid, "Исторический синтетический контекст")
    async with sessions() as session, session.begin():
        row = await session.scalar(select(Message).where(Message.role == "user"))
        row.created_at = utcnow() - timedelta(days=400)
        dialogue = await session.scalar(select(Conversation))
        dialogue.context_summary = "Пересказ удаляемого контекста"
        dialogue.context_summary_until = utcnow() - timedelta(days=1)
        dialogue.context_summary_messages = 1
    await prune(sessions, 365)
    async with sessions() as session:
        dialogue = await session.scalar(select(Conversation))
        assert not dialogue.context_summary and dialogue.context_summary_until is None
        assert await session.scalar(select(PromptTrace)) is None
        assert len(list((await session.scalars(select(Message))).all())) == 1


async def test_active_generation_sources_are_not_pruned_until_it_finishes(client, auth, sessions):
    from emo_core.db.models import Message, PromptTrace

    cid = await conversation(client, auth)
    await send(client, auth, cid, "Синтетический контекст")
    async with sessions() as session, session.begin():
        row = await session.scalar(select(Message).where(Message.role == "user"))
        row.created_at = utcnow() - timedelta(days=400)
        row.details = {"processing": "pending", "processing_started_at": utcnow().isoformat()}
    assert await prune(sessions, 365) == 0
    async with sessions() as session, session.begin():
        assert await session.scalar(select(PromptTrace)) is not None
        row = await session.scalar(select(Message).where(Message.role == "user"))
        row.details = {"processing": "interrupted"}
    assert await prune(sessions, 365) >= 1


async def test_retention_removes_expired_only(client, auth, sessions):
    """Проверяет удаление только просроченных записей и сохранение свежих данных."""
    for name in ("anna", "maria"):
        await client.post(
            "/api/v1/emo/checkins", headers=auth[name], json={"mood_score": 5, "anxiety_score": 2}
        )
    async with sessions() as s, s.begin():
        row = await s.scalar(select(Checkin).order_by(Checkin.created_at))
        row.created_at = utcnow() - timedelta(days=400)
    assert await prune(sessions, 365) == 1
    async with sessions() as s:
        rows = list((await s.scalars(select(Checkin))).all())
        assert len(rows) == 1

from datetime import timedelta

from sqlalchemy import delete, select

from emo_core.db import models as m
from emo_core.db.base import utcnow


async def prune(sessions, days: int, now=None):
    """Удаляет истёкшие данные пользователей под блокировкой профиля.

    Граница — now минус days суток. Занятые профили пропускает через SKIP LOCKED,
    память очищает от выдержек удалённых сообщений. Возвращаемый счётчик включает
    сообщения, самооценки, попытки, safety-события и жалобы, но не все каскадные
    удаления. Записи дедупликации заданий сохраняет; резервные копии не обрабатывает.
    """
    cutoff = (now or utcnow()) - timedelta(days=days)
    removed = 0
    async with sessions() as session:
        ids = list((await session.scalars(select(m.UserSettings.user_id))).all())
    for user_id in ids:
        async with sessions() as session, session.begin():
            user = await session.scalar(
                select(m.UserSettings)
                .where(m.UserSettings.user_id == user_id)
                .with_for_update(skip_locked=True)
            )
            if user is None:
                continue
            old_messages = list(
                (
                    await session.scalars(
                        select(m.Message.id).where(
                            m.Message.user_id == user_id, m.Message.created_at < cutoff
                        )
                    )
                ).all()
            )
            for model in (m.NotificationOutbox, m.SupportFeedback):
                await session.execute(
                    delete(model).where(
                        model.user_id == user_id, model.message_id.in_(old_messages)
                    )
                )
            await session.execute(
                delete(m.MessageSignal).where(
                    m.MessageSignal.user_id == user_id, m.MessageSignal.observed_at < cutoff
                )
            )
            await session.execute(
                delete(m.Message).where(m.Message.user_id == user_id, m.Message.created_at < cutoff)
            )
            for model, column in (
                (m.Checkin, m.Checkin.created_at),
                (m.TestAttempt, m.TestAttempt.started_at),
                (m.SafetyEvent, m.SafetyEvent.created_at),
                (m.Complaint, m.Complaint.last_seen_at),
            ):
                result = await session.execute(
                    delete(model).where(model.user_id == user_id, column < cutoff)
                )
                removed += result.rowcount
            memory = await session.get(m.UserState, user_id)
            if memory:
                memory.facts = [
                    f for f in memory.facts if f.get("observed_at", "") >= cutoff.isoformat()
                ]
                if memory.updated_at < cutoff:
                    memory.summary, memory.recurring_topics, memory.facts = "", [], []
                elif old_messages:
                    memory.summary = "\n".join(f["text"] for f in memory.facts)[-3000:]
            removed += len(old_messages)
    return removed

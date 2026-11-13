from datetime import timedelta

from sqlalchemy import delete, select, update

from emo_core.db import models as m
from emo_core.db.base import utcnow


async def prune(sessions, days: int, now=None, *, prompt_days=7):
    """Удаляет истёкшие данные пользователей под блокировкой профиля.

    Граница — now минус days суток. Занятые профили пропускает через SKIP LOCKED,
    память очищает от выдержек удалённых сообщений. Возвращаемый счётчик включает
    сообщения, самооценки, попытки, safety-события и жалобы, но не все каскадные
    удаления. Записи дедупликации заданий сохраняет; резервные копии не обрабатывает.
    """
    now = now or utcnow()
    cutoff = now - timedelta(days=days)
    prompt_cutoff = now - timedelta(days=min(days, prompt_days))
    removed = 0
    after = None
    while True:
        async with sessions() as session:
            ids = list(
                (
                    await session.scalars(
                        select(m.UserSettings.user_id)
                        .where(m.UserSettings.user_id > after if after else True)
                        .order_by(m.UserSettings.user_id)
                        .limit(200)
                    )
                ).all()
            )
        if not ids:
            break
        for user_id in ids:
            removed += await _prune_user(sessions, user_id, cutoff, prompt_cutoff, now)
        after = ids[-1]
    return removed


async def _prune_user(sessions, user_id, cutoff, prompt_cutoff, now):
    removed = 0
    # Keep selection/deletion under the same lock as chat; skip active writers.
    async with sessions() as session, session.begin():
        user = await session.scalar(
            select(m.UserSettings)
            .where(m.UserSettings.user_id == user_id)
            .with_for_update(skip_locked=True)
        )
        if user is None:
            return 0
        # Generations use a snapshot outside the lock. Do not purge its sources
        # while a live request may still publish them again. The longest bounded
        # chat lease is below one hour; abandoned requests cannot block forever.
        active = await session.scalar(
            select(m.Message.id)
            .where(
                m.Message.user_id == user_id,
                m.Message.details["processing"].as_string() == "pending",
                m.Message.details["processing_started_at"].as_string()
                > (now - timedelta(hours=1)).isoformat(),
            )
            .limit(1)
        )
        if active:
            return 0
        old_messages = select(m.Message.id).where(
            m.Message.user_id == user_id, m.Message.created_at < cutoff
        )
        has_old_messages = await session.scalar(old_messages.limit(1)) is not None
        if has_old_messages:
            # Derived summaries and newer prompt snapshots may quote expired sources.
            await session.execute(
                update(m.Conversation)
                .where(m.Conversation.user_id == user_id)
                .values(context_summary="", context_summary_until=None, context_summary_messages=0)
            )
        await session.execute(
            delete(m.PromptTrace).where(
                m.PromptTrace.user_id == user_id,
                True if has_old_messages else m.PromptTrace.created_at < prompt_cutoff,
            )
        )
        for model in (m.NotificationOutbox, m.SupportFeedback):
            await session.execute(
                delete(model).where(model.user_id == user_id, model.message_id.in_(old_messages))
            )
        await session.execute(
            delete(m.MessageSignal).where(
                m.MessageSignal.user_id == user_id, m.MessageSignal.observed_at < cutoff
            )
        )
        result = await session.execute(
            delete(m.Message).where(m.Message.user_id == user_id, m.Message.created_at < cutoff)
        )
        removed += result.rowcount
        for model, column in (
            (m.VoiceInput, m.VoiceInput.created_at),
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
            elif has_old_messages:
                memory.summary = "\n".join(f["text"] for f in memory.facts)[-3000:]
    return removed

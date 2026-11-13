import asyncio
import logging

from sqlalchemy import select

from emo_core.core.config import get_settings
from emo_core.core.logging import configure_logging
from emo_core.db.models import NotificationOutbox, UserSettings
from emo_core.db.session import Session, engine
from emo_core.providers.notification_provider import GENERIC_PUSH, InAppOnlyProvider
from emo_core.services.proactive_engine import ProactiveMessagingEngine
from emo_core.services.retention import prune


async def dispatch_outbox(sessions, provider):
    """Обрабатывает ожидающие push-уведомления с блокировкой профиля и записей outbox.

    За профиль берёт до 50 записей с числом попыток меньше пяти. При отключённом
    push отменяет доставку; InAppOnlyProvider помечает not_configured. Внешнему
    провайдеру передаёт общий текст и message_id как ключ повтора; ошибки сохраняет
    фиксированным кодом, после пятой попытки — статус failed.
    """
    async with sessions() as session:
        users = list(
            (
                await session.scalars(
                    select(NotificationOutbox.user_id)
                    .where(NotificationOutbox.status == "pending", NotificationOutbox.attempts < 5)
                    .distinct()
                )
            ).all()
        )
    for user_id in users:
        async with sessions() as session, session.begin():
            # Same lock order as chat, deletion and proactive delivery.
            user = await session.scalar(
                select(UserSettings)
                .where(UserSettings.user_id == user_id)
                .with_for_update(skip_locked=True)
            )
            if user is None:
                continue
            entries = list(
                (
                    await session.scalars(
                        select(NotificationOutbox)
                        .where(
                            NotificationOutbox.user_id == user_id,
                            NotificationOutbox.status == "pending",
                            NotificationOutbox.attempts < 5,
                        )
                        .with_for_update(skip_locked=True)
                        .limit(50)
                    )
                ).all()
            )
            for entry in entries:
                if user.deleted_at or not user.push_enabled:
                    entry.status = "cancelled"
                    continue
                entry.attempts += 1
                if isinstance(provider, InAppOnlyProvider):
                    entry.status = "not_configured"
                    continue
                try:
                    await provider.send(entry.user_id, GENERIC_PUSH, str(entry.message_id))
                    entry.status = "sent"
                except Exception:
                    entry.last_error = "notification_provider_failed"
                    if entry.attempts >= 5:
                        entry.status = "failed"


async def main():
    """Запускает периодический цикл планировщика и доставки outbox.

    Интервал задаёт proactive_poll_seconds. Ошибка цикла записывается без
    пользовательских данных; при выходе освобождает пул БД.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    scheduler = ProactiveMessagingEngine(settings, Session)
    next_prune = 0
    try:
        while True:
            if asyncio.get_running_loop().time() >= next_prune:
                try:
                    await prune(
                        Session,
                        settings.retention_days,
                        prompt_days=settings.prompt_trace_retention_days,
                    )
                    next_prune = asyncio.get_running_loop().time() + 86400
                except Exception:
                    logging.getLogger(__name__).error("retention_cycle_failed")
            try:
                await scheduler.run_once()
                await dispatch_outbox(Session, InAppOnlyProvider())
            except Exception:
                logging.getLogger(__name__).error("worker_cycle_failed")
            await asyncio.sleep(settings.proactive_poll_seconds)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

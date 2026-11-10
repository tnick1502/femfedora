from datetime import UTC, datetime, time, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select

from emo_core.core.auth import DEMO_USERS
from emo_core.db.models import Message, SafetyEvent, UserSettings
from emo_core.services.proactive_engine import ProactiveMessagingEngine
from tests.conftest import conversation

NOW = datetime(2026, 11, 10, 12, tzinfo=UTC)


async def prepare(client, auth, sessions, now=NOW):
    """Создаёт профиль, беседу и исходные условия для цикла инициативных сообщений."""
    cid = await conversation(client, auth)
    await client.patch(
        "/api/v1/emo/settings",
        headers=auth["anna"],
        json={"proactive_enabled": True, "timezone": "UTC"},
    )
    async with sessions() as s, s.begin():
        user = await s.get(UserSettings, DEMO_USERS["anna"])
        user.created_at = now - timedelta(days=10)
    return UUID(cid)


async def test_consent_and_delivery_idempotence(client, auth, sessions, config):
    """Проверяет обязательное согласие и однократность отправки инициативного сообщения."""
    cid = await conversation(client, auth)
    worker = ProactiveMessagingEngine(config, sessions)
    assert (await worker.run_once(NOW))["sent"] == 0
    await client.patch(
        "/api/v1/emo/settings",
        headers=auth["anna"],
        json={"proactive_enabled": True, "timezone": "UTC"},
    )
    assert (await worker.run_once(NOW))["sent"] == 1
    assert (await worker.run_once(NOW))["sent"] == 0
    data = (
        await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
    ).json()["items"]
    assert len(data) == 1 and data[0]["origin"] == "proactive" and data[0]["role"] == "assistant"


@pytest.mark.parametrize("hour", [7, 9, 19, 23])
async def test_allowed_hours(client, auth, sessions, config, hour):
    """Проверяет запрет отправки вне разрешённого времени пользователя."""
    await prepare(client, auth, sessions)
    worker = ProactiveMessagingEngine(config, sessions)
    assert (await worker.run_once(NOW.replace(hour=hour)))["sent"] == 0


async def test_custom_quiet_hours(client, auth, sessions, config):
    """Проверяет запрет отправки в личные тихие часы."""
    await prepare(client, auth, sessions)
    await client.patch(
        "/api/v1/emo/settings",
        headers=auth["anna"],
        json={"quiet_hours_start": "11:00:00", "quiet_hours_end": "13:00:00"},
    )
    assert (await ProactiveMessagingEngine(config, sessions).run_once(NOW))["sent"] == 0


async def test_recent_activity_suppresses(client, auth, sessions, config):
    """Проверяет подавление инициативного контакта после недавней активности."""
    cid = await prepare(client, auth, sessions)
    async with sessions() as s, s.begin():
        s.add(
            Message(
                user_id=DEMO_USERS["anna"],
                conversation_id=cid,
                role="user",
                origin="user_input",
                content="Привет",
                created_at=NOW - timedelta(hours=3),
            )
        )
    assert (await ProactiveMessagingEngine(config, sessions).run_once(NOW))["sent"] == 0


async def test_unread_and_two_unanswered_pause(client, auth, sessions, config):
    """Проверяет ожидание прочтения и автоматическую паузу после двух контактов без ответа."""
    cid = await prepare(client, auth, sessions)
    worker = ProactiveMessagingEngine(config, sessions)
    assert (await worker.run_once(NOW))["sent"] == 1
    assert (await worker.run_once(NOW + timedelta(days=8)))["sent"] == 0
    # Reading is different from replying: second contact can be read yet unanswered.
    await client.post(f"/api/v1/emo/conversations/{cid}/read", headers=auth["anna"])
    assert (await worker.run_once(NOW + timedelta(days=8, hours=1)))["sent"] == 1
    await worker.run_once(NOW + timedelta(days=16))
    async with sessions() as s:
        user = await s.get(UserSettings, DEMO_USERS["anna"])
        assert user.proactive_paused_until == NOW + timedelta(days=46)


async def test_safety_suppresses_and_no_ordinary_followups(client, auth, sessions, config):
    """Проверяет подавление обычных инициативных сообщений после срочного маршрута."""
    await prepare(client, auth, sessions)
    async with sessions() as s, s.begin():
        s.add(
            SafetyEvent(
                user_id=DEMO_USERS["anna"],
                risk_type="possible_danger",
                risk_route="urgent",
                action_taken="urgent_template",
                policy_version="1",
                created_at=NOW,
            )
        )
    assert (await ProactiveMessagingEngine(config, sessions).run_once(NOW))["sent"] == 0


async def test_priority_and_push_outbox(client, auth, sessions, config):
    """Проверяет приоритет согласованного контакта и создание push-записи outbox."""
    cid = await prepare(client, auth, sessions)
    await client.patch("/api/v1/emo/settings", headers=auth["anna"], json={"push_enabled": True})
    worker = ProactiveMessagingEngine(config, sessions)
    async with sessions() as s, s.begin():
        await worker.enqueue(
            s,
            user_id=DEMO_USERS["anna"],
            conversation_id=cid,
            job_type="agreed_followup",
            deduplication_key="test-followup",
            scheduled_for=NOW,
            details={"explicit_consent": True},
        )
    assert (await worker.run_once(NOW))["sent"] == 1
    async with sessions() as s:
        m = await s.scalar(select(Message).where(Message.origin == "proactive"))
        assert m.details["job_type"] == "agreed_followup"
    assert (await worker.run_once(NOW))["sent"] == 0


async def test_gap_and_rate_limits(client, auth, sessions, config):
    """Проверяет минимальный интервал и ограничения числа инициативных сообщений."""
    cid = await prepare(client, auth, sessions)
    worker = ProactiveMessagingEngine(config, sessions)
    async with sessions() as s, s.begin():
        user = await s.get(UserSettings, DEMO_USERS["anna"])
        for day in [2, 5]:
            s.add(
                Message(
                    user_id=user.user_id,
                    conversation_id=cid,
                    role="assistant",
                    origin="proactive",
                    content="test",
                    created_at=NOW - timedelta(days=day),
                    read_at=NOW - timedelta(days=1),
                )
            )
        # Reply after both contacts avoids unanswered pause but older than activity suppression.
        s.add(
            Message(
                user_id=user.user_id,
                conversation_id=cid,
                role="user",
                origin="user_input",
                content="test",
                created_at=NOW - timedelta(hours=30),
            )
        )
    assert (await worker.run_once(NOW))["sent"] == 0


def test_timezone_dst_and_changes(config):
    """Проверяет локальные часы с учётом DST и смены часового пояса."""
    worker = ProactiveMessagingEngine(config, None)
    user = UserSettings(
        timezone="America/New_York", quiet_hours_start=time(20), quiet_hours_end=time(9)
    )
    assert worker.allowed_time(user, datetime(2026, 3, 8, 14, tzinfo=UTC))  # 10 EDT
    assert not worker.allowed_time(user, datetime(2026, 3, 7, 14, tzinfo=UTC))  # 9 EST
    assert worker.allowed_time(user, datetime(2026, 11, 1, 15, tzinfo=UTC))  # 10 EST
    user.timezone = "Asia/Tokyo"
    assert not worker.allowed_time(user, datetime(2026, 11, 1, 15, tzinfo=UTC))
    user.timezone = "invalid"
    assert not worker.allowed_time(user, NOW)


async def test_notification_failure_never_duplicates_message(client, auth, sessions, config):
    """Проверяет повтор доставки push без дублирования сообщения в беседе."""
    from emo_core.db.models import NotificationOutbox
    from emo_core.workers.proactive_worker import dispatch_outbox

    cid = await prepare(client, auth, sessions)
    await client.patch("/api/v1/emo/settings", headers=auth["anna"], json={"push_enabled": True})
    worker = ProactiveMessagingEngine(config, sessions)
    await worker.run_once(NOW)

    class Broken:
        async def send(self, user_id, text, idempotency_key):
            """Проверяет общий текст push и имитирует сбой внешней доставки."""
            assert text == "У вас новое сообщение в приложении"
            raise RuntimeError("secret provider body")

    await dispatch_outbox(sessions, Broken())
    await dispatch_outbox(sessions, Broken())
    async with sessions() as s:
        row = await s.scalar(select(NotificationOutbox))
        assert row.attempts == 2 and row.last_error == "notification_provider_failed"
    assert (await worker.run_once(NOW))["sent"] == 0
    assert (
        len(
            (
                await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
            ).json()["items"]
        )
        == 1
    )

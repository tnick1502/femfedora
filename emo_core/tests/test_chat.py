from uuid import uuid4

from emo_core.core.exceptions import ProviderUnavailable
from tests.conftest import conversation, send


async def test_whole_chat_deadline_releases_capacity_and_marks_interrupted(
    client, auth, app, monkeypatch
):
    import asyncio
    from datetime import timedelta

    from emo_core.services.chat_engine import ChatEngine

    original = ChatEngine.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.processing_timeout = timedelta(seconds=0.1)

    monkeypatch.setattr(ChatEngine, "__init__", initialize)

    async def generate(messages, repair=False):
        await asyncio.Event().wait()

    app.state.llm.generate = generate
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Привет")
    assert response.status_code == 503
    history = (
        await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
    ).json()["items"]
    assert history[0]["metadata"]["processing"] == "interrupted"
    assert not app.state.chat_slots.locked()


async def test_unchanged_poll_omits_reports_and_deletion_invalidates_page_version(
    client, auth, sessions
):
    from uuid import UUID

    from sqlalchemy import delete

    from emo_core.db.models import Message

    cid = await conversation(client, auth)
    await send(client, auth, cid, "Привет")
    path = f"/api/v1/emo/conversations/{cid}/messages?limit=100"
    first = (await client.get(path, headers=auth["anna"])).json()
    poll = (
        await client.get(path + "&known_version=" + first["version"], headers=auth["anna"])
    ).json()
    assert poll["unchanged"] and poll["items"] == []
    assert "decision_report" not in str(poll)
    async with sessions() as session, session.begin():
        await session.execute(delete(Message).where(Message.id == UUID(first["items"][-1]["id"])))
    changed = (
        await client.get(path + "&known_version=" + first["version"], headers=auth["anna"])
    ).json()
    assert changed["version"] != first["version"] and len(changed["items"]) == 1
    assert (
        await client.get(path + "&known_version=" + changed["version"], headers=auth["maria"])
    ).status_code == 404


async def test_budget_blocks_new_work_but_replays_and_urgent_templates_still_work(
    client, auth, config, app
):
    config.chat_messages_per_hour = 1
    cid, key = await conversation(client, auth), str(uuid4())
    first = await send(client, auth, cid, "Привет", client_message_id=key)
    assert first.status_code == 200
    assert (await send(client, auth, cid, "Ещё одно сообщение")).status_code == 429
    assert (await send(client, auth, cid, "Привет", client_message_id=key)).json()["replayed"]
    # Ordinary work is saturated as well; explicit urgent templates bypass model capacity.
    for _ in range(config.chat_concurrency):
        await app.state.chat_slots.acquire()
    try:
        urgent = await send(client, auth, cid, "Я хочу убить себя сейчас")
        assert urgent.status_code == 200
        assert urgent.json()["assistant_message"]["origin"] == "safety_template"
    finally:
        for _ in range(config.chat_concurrency):
            app.state.chat_slots.release()


async def test_urgent_message_supersedes_an_ordinary_pending_generation(client, auth, app):
    import asyncio
    from unittest.mock import AsyncMock

    entered, release = asyncio.Event(), asyncio.Event()

    async def generate(messages, repair=False):
        entered.set()
        await release.wait()
        return "Привет! О чём хочется поговорить?"

    app.state.llm.generate = AsyncMock(side_effect=generate)
    cid, key = await conversation(client, auth), str(uuid4())
    normal = asyncio.create_task(send(client, auth, cid, "Привет", client_message_id=key))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        urgent = await send(client, auth, cid, "Я хочу убить себя сейчас")
        assert (
            urgent.status_code == 200
            and urgent.json()["assistant_message"]["origin"] == "safety_template"
        )
        release.set()
        assert (await normal).status_code == 409
        assert (await send(client, auth, cid, "Привет", client_message_id=key)).status_code == 409
        history = (
            await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
        ).json()["items"]
        assert len([m for m in history if m["role"] == "assistant"]) == 1
        assert app.state.llm.generate.await_count == 1
    finally:
        release.set()
        await normal


async def test_chat_capacity_rejects_without_calling_models(client, auth, app, config):
    from unittest.mock import AsyncMock

    cid = await conversation(client, auth)
    app.state.llm.generate = AsyncMock(side_effect=AssertionError("must not call"))
    for _ in range(config.chat_concurrency):
        await app.state.chat_slots.acquire()
    try:
        assert (await send(client, auth, cid, "Привет")).status_code == 429
    finally:
        for _ in range(config.chat_concurrency):
            app.state.chat_slots.release()
    app.state.llm.generate.assert_not_awaited()


async def test_stale_generation_cannot_publish_or_interrupt_its_replacement(
    client, auth, app, sessions
):
    import asyncio
    from datetime import timedelta

    from sqlalchemy import select

    from emo_core.db.base import utcnow
    from emo_core.db.models import Message

    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def generate(messages, repair=False):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await release[index].wait()
        return "Спасибо, что написала. Что хочется обсудить?"

    app.state.llm.generate = generate
    cid, key = await conversation(client, auth), str(uuid4())
    first = asyncio.create_task(send(client, auth, cid, "Мне тревожно", client_message_id=key))
    replacement = None
    try:
        await asyncio.wait_for(entered[0].wait(), 5)
        async with sessions() as session, session.begin():
            row = await session.scalar(select(Message).where(Message.role == "user"))
            row.details = {
                **row.details,
                "processing_started_at": (utcnow() - timedelta(hours=1)).isoformat(),
            }
            old_lease = row.details["processing_lease"]
        replacement = asyncio.create_task(
            send(client, auth, cid, "Мне тревожно", client_message_id=key)
        )
        await asyncio.wait_for(entered[1].wait(), 5)
        release[0].set()
        assert (await first).status_code == 409
        async with sessions() as session:
            row = await session.scalar(select(Message).where(Message.role == "user"))
            assert row.details["processing"] == "pending"
            assert row.details["processing_lease"] != old_lease
        release[1].set()
        assert (await replacement).status_code == 200
        assert calls == 2
        async with sessions() as session:
            assert len(list((await session.scalars(select(Message))).all())) == 2
    finally:
        for event in release:
            event.set()
        await asyncio.gather(
            *(task for task in (first, replacement) if task), return_exceptions=True
        )


async def test_level_cancellation_cleans_up_pending_chat(client, auth, app, sessions):
    import asyncio

    import anyio
    from sqlalchemy import select

    from emo_core.db.models import Message

    entered = asyncio.Event()

    async def generate(messages, repair=False):
        entered.set()
        await asyncio.Event().wait()

    app.state.llm.generate = generate
    cid = await conversation(client, auth)
    scopes = []

    async def request():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await send(client, auth, cid, "Мне тревожно")

    task = asyncio.create_task(request())
    await asyncio.wait_for(entered.wait(), 5)
    scopes[0].cancel()
    await asyncio.wait_for(task, 5)
    async with sessions() as session:
        row = await session.scalar(select(Message).where(Message.role == "user"))
        assert row.details["processing"] == "interrupted"


async def test_waiting_retry_releases_connection_before_resuming(
    client, auth, app, sessions, config
):
    import asyncio

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from emo_core.db.models import Message
    from emo_core.schemas.chat import MessageCreate
    from emo_core.services.chat_engine import ChatEngine

    original = app.state.llm.generate

    async def fail(messages, repair=False):
        raise RuntimeError("interrupted synthetic request")

    app.state.llm.generate = fail
    cid, key = await conversation(client, auth), uuid4()
    assert (
        await send(client, auth, cid, "Мне тревожно", client_message_id=str(key))
    ).status_code == 500
    app.state.llm.generate = original
    async with sessions() as session:
        row = await session.scalar(select(Message).where(Message.role == "user"))
        uid, mid, conversation_id = row.user_id, row.id, row.conversation_id
    engine = create_async_engine(
        sessions.kw["bind"].url, pool_size=1, max_overflow=0, pool_timeout=0.4
    )
    try:
        one = async_sessionmaker(engine, expire_on_commit=False)
        result = await asyncio.wait_for(
            ChatEngine(config, app.state.llm, app.state.jev)._await_result(
                one,
                uid,
                conversation_id,
                MessageCreate(content="Мне тревожно", client_message_id=key),
                mid,
            ),
            3,
        )
        assert result["assistant_message"]["role"] == "assistant"
    finally:
        await engine.dispose()


async def test_persistence_and_idempotency(client, auth):
    """Проверяет сохранение обеих реплик и возврат прежнего ответа при повторе UUID."""
    cid = await conversation(client, auth)
    mid = str(uuid4())
    first = await send(client, auth, cid, "Мне тревожно", client_message_id=mid)
    assert first.status_code == 200, first.text
    second = await send(client, auth, cid, "Мне тревожно", client_message_id=mid)
    assert second.json()["replayed"] is True
    assert first.json()["assistant_message"]["id"] == second.json()["assistant_message"]["id"]
    history = (
        await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
    ).json()["items"]
    assert len(history) == 2
    assert (await send(client, auth, cid, "другой текст", client_message_id=mid)).status_code == 409


async def test_context_and_provider_replacement(client, auth, app):
    """Проверяет передачу истории и памяти заменённому LLM-провайдеру."""

    class Spy:
        def __init__(self):
            """Создаёт список записанных запросов к тестовой модели."""
            self.prompts = []

        async def generate(self, messages, repair=False):
            """Запоминает переданный контекст и возвращает тестовый ответ."""
            self.prompts.append(messages)
            return "Спасибо, что рассказываешь. Что хотелось бы обсудить?"

    app.state.llm = spy = Spy()
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Я устала и хочу поддержки")
    await send(client, auth, cid, "Продолжим")
    assert any("устала" in m["content"] for m in spy.prompts[-1])
    other = await conversation(client, auth)
    await send(client, auth, other, "Привет")
    assert "устала" in spy.prompts[-1][1]["content"]
    assert len(spy.prompts[-1]) <= 20


async def test_llm_failure_is_saved(client, auth, app):
    """Проверяет сохранение понятного шаблонного ответа при недоступности LLM."""

    class Broken:
        async def generate(self, messages, repair=False):
            """Имитирует недоступность LLM при генерации."""
            raise ProviderUnavailable("llm_unavailable")

    app.state.llm = Broken()
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Привет")
    assert response.status_code == 200
    answer = response.json()["assistant_message"]
    assert answer["origin"] == "safety_template"
    assert "временно недоступен" in answer["content"]


async def test_interrupted_processing_resumes_without_duplicates(client, auth, app):
    """Проверяет возобновление прерванного сообщения по тому же UUID без дублей."""

    class Crash:
        async def generate(self, messages, repair=False):
            """Прерывает обработку непредусмотренной ошибкой тестовой модели."""
            raise RuntimeError("unexpected provider crash")

    app.state.llm = Crash()
    cid = await conversation(client, auth)
    mid = str(uuid4())
    assert (await send(client, auth, cid, "Привет", client_message_id=mid)).status_code == 500
    history = (
        await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
    ).json()["items"]
    assert [m["role"] for m in history] == ["user"]
    assert history[0]["metadata"]["processing"] == "interrupted"

    class Works:
        async def generate(self, messages, repair=False):
            """Возвращает ответ при повторной обработке прерванного сообщения."""
            return "Спасибо, что написала. Что хочется обсудить?"

    app.state.llm = Works()
    retry = await send(client, auth, cid, "Привет", client_message_id=mid)
    assert retry.status_code == 200 and retry.json()["replayed"] is False
    history = (
        await client.get(f"/api/v1/emo/conversations/{cid}/messages", headers=auth["anna"])
    ).json()["items"]
    assert [m["role"] for m in history] == ["assistant", "user"]
    assert "processing" not in history[1]["metadata"]


async def test_new_message_waits_for_previous_reply(client, auth, app):
    """Проверяет конфликт параллельной реплики при доступности чтения настроек."""
    import asyncio

    gate, entered = asyncio.Event(), asyncio.Event()

    class Slow:
        async def generate(self, messages, repair=False):
            """Удерживает генерацию до сигнала теста, чтобы проверить параллельные запросы."""
            entered.set()
            await gate.wait()
            return "Спасибо, что написала. Что хочется обсудить?"

    app.state.llm = Slow()
    cid = await conversation(client, auth)
    mid = str(uuid4())
    first = asyncio.create_task(send(client, auth, cid, "Первое", client_message_id=mid))
    await entered.wait()
    # Пока идёт генерация: чтение и изменение настроек не блокируются, другой текст отклоняется,
    # повтор того же сообщения дожидается единственного ответа.
    assert (await client.get("/api/v1/emo/settings", headers=auth["anna"])).status_code == 200
    assert (
        await client.patch(
            "/api/v1/emo/settings", headers=auth["anna"], json={"timezone": "Europe/Moscow"}
        )
    ).status_code == 200
    other = await send(client, auth, cid, "Второе")
    assert other.status_code == 409
    assert other.json()["detail"]["code"] == "previous_message_processing"
    duplicate = asyncio.create_task(send(client, auth, cid, "Первое", client_message_id=mid))
    await asyncio.sleep(0.4)
    gate.set()
    a, b = (await first).json(), (await duplicate).json()
    assert a["assistant_message"]["id"] == b["assistant_message"]["id"]
    assert sorted([a["replayed"], b["replayed"]]) == [False, True]
    assert (await send(client, auth, cid, "Второе")).status_code == 200

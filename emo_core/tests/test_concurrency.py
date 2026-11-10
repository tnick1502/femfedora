import asyncio
import os
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select

from emo_core.db.models import Message
from emo_core.services.proactive_engine import ProactiveMessagingEngine
from tests.conftest import conversation, send
from tests.test_proactive import NOW, prepare
from tests.test_questionnaire_workflow import start

postgres = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="PostgreSQL row locks require TEST_DATABASE_URL",
)


@postgres
@pytest.mark.postgres
async def test_two_workers_same_job_once(client, auth, sessions, config):
    """Проверяет однократную отправку задания двумя конкурентными PostgreSQL-worker."""
    await prepare(client, auth, sessions)
    workers = [ProactiveMessagingEngine(config, sessions) for _ in range(2)]
    results = await asyncio.gather(*(w.run_once(NOW) for w in workers))
    assert sum(r["sent"] for r in results) == 1
    async with sessions() as s:
        assert (
            await s.scalar(
                select(func.count()).select_from(Message).where(Message.origin == "proactive")
            )
            == 1
        )


@postgres
@pytest.mark.postgres
async def test_concurrent_idempotent_chat(client, auth):
    """Проверяет один ответ на параллельные запросы чата с одинаковым UUID."""
    cid = await conversation(client, auth)
    mid = str(uuid4())
    results = await asyncio.gather(
        *(send(client, auth, cid, "Привет", client_message_id=mid) for _ in range(3))
    )
    assert all(r.status_code == 200 for r in results)
    assert len({r.json()["assistant_message"]["id"] for r in results}) == 1


@postgres
@pytest.mark.postgres
async def test_parallel_users(client, auth):
    """Проверяет независимую обработку сообщений разных пользователей."""
    a, b = await asyncio.gather(conversation(client, auth), conversation(client, auth, "maria"))
    results = await asyncio.gather(
        send(client, auth, a, "Мне тревожно"), send(client, auth, b, "Мне одиноко", "maria")
    )
    assert all(r.status_code == 200 for r in results)
    assert (
        results[0].json()["assistant_message"]["user_id"]
        != results[1].json()["assistant_message"]["user_id"]
    )


@postgres
@pytest.mark.postgres
async def test_questionnaire_parallel_start_is_idempotent(client, auth):
    """Проверяет одно прохождение при одновременном старте с одним client_request_id."""
    key = str(uuid4())
    attempts = await asyncio.gather(*(start(client, auth, request_id=key) for _ in range(3)))
    assert len({a["id"] for a in attempts}) == 1


@postgres
@pytest.mark.postgres
async def test_questionnaire_parallel_edit_detects_stale_revision(client, auth):
    """Проверяет конфликт одной из двух одновременных правок одной ревизии черновика."""
    a = await start(client, auth)
    path = "/api/v1/emo/questionnaire-attempts/" + a["id"]
    results = await asyncio.gather(
        *(
            client.patch(
                path, headers=auth["anna"], json={"expected_revision": 0, "answers": {"1": answer}}
            )
            for answer in (1, 2)
        )
    )
    assert sorted(r.status_code for r in results) == [200, 409]
    draft = (await client.get(path, headers=auth["anna"])).json()
    assert draft["revision"] == 1
    assert draft["answers"]["1"] in (1, 2)


@postgres
@pytest.mark.postgres
async def test_questionnaire_parallel_completion_preserves_first_result(client, auth):
    """Проверяет одинаковый сохранённый результат повторных одновременных завершений."""
    a = await start(client, auth)
    path = "/api/v1/emo/questionnaire-attempts/" + a["id"]
    results = await asyncio.gather(
        *(
            client.post(
                path + "/complete",
                headers=auth["anna"],
                json={"expected_revision": 0, "answers": {"1": 1, "2": 2}},
            )
            for _ in range(3)
        )
    )
    assert all(r.status_code == 200 for r in results)
    assert all(r.json() == results[0].json() for r in results)


@postgres
@pytest.mark.postgres
async def test_model_call_holds_no_connection_and_no_user_lock(config, auth, client):
    """Проверяет освобождение соединения и блокировки профиля на время вызова модели."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from emo_core.main import create_app

    engine = create_async_engine(
        os.environ["TEST_DATABASE_URL"], pool_size=1, max_overflow=0, pool_timeout=3
    )
    single = async_sessionmaker(engine, expire_on_commit=False)
    seen = {}

    class Slow:
        async def generate(self, messages, repair=False):
            # Единственное соединение пула должно быть свободно, строка пользователя — не заблокирована.
            """Проверяет возможность взять соединение и блокировку профиля во время генерации."""
            async with single() as s, s.begin():
                await s.execute(
                    text("SELECT 1 FROM emo_user_settings FOR UPDATE NOWAIT"),
                )
            seen["free"] = True
            return "Спасибо, что написала. Что хочется обсудить?"

    app = create_app(config, single, llm=Slow())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as one:
        cid = (
            await one.post("/api/v1/emo/conversations", headers=auth["anna"], json={"title": "t"})
        ).json()["id"]
        response = await one.post(
            f"/api/v1/emo/conversations/{cid}/messages",
            headers=auth["anna"],
            json={"content": "Привет", "client_message_id": str(uuid4())},
        )
    await engine.dispose()
    assert response.status_code == 200 and seen == {"free": True}
    assert response.json()["assistant_message"]["origin"] == "reactive"

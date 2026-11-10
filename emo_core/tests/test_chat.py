from uuid import uuid4

from emo_core.core.exceptions import ProviderUnavailable
from tests.conftest import conversation, send


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

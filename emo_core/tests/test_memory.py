from tests.conftest import conversation, send


async def test_model_history_query_excludes_diagnostic_json(client, auth, sessions):
    from uuid import UUID

    from sqlalchemy import event, select

    from emo_core.db.models import Message
    from emo_core.services.memory_manager import MemoryManager

    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно")
    async with sessions() as session, session.begin():
        message = await session.scalar(select(Message).where(Message.role == "assistant"))
        uid = message.user_id
        message.details = {"synthetic_diagnostic": "x" * 240_000}
    statements = []
    engine = sessions.kw["bind"].sync_engine

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        async with sessions() as session:
            history = await MemoryManager().history(session, uid, UUID(cid))
        assert len(history) == 2 and history[0]["content"] == "Мне тревожно"
        assert all("metadata" not in statement for statement in statements)
    finally:
        event.remove(engine, "before_cursor_execute", capture)


async def test_inspect_correct_delete(client, auth):
    """Проверяет просмотр, исправление и удаление пользовательской памяти через API."""
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне помогает разговор с сестрой")
    state = (await client.get("/api/v1/emo/state", headers=auth["anna"])).json()
    assert "сестрой" in state["summary"]
    assert state["facts"][0]["source"] == "user_report"
    await client.patch(
        "/api/v1/emo/memory",
        headers=auth["anna"],
        json={"summary": "Исправленный контекст", "preferences": {"style": "кратко"}},
    )
    assert (await client.get("/api/v1/emo/state", headers=auth["anna"])).json()[
        "summary"
    ] == "Исправленный контекст"
    await client.delete("/api/v1/emo/memory", headers=auth["anna"])
    assert (await client.get("/api/v1/emo/state", headers=auth["anna"])).json()["summary"] == ""


async def test_deleted_memory_not_rebuilt_from_old_history(client, auth):
    """Проверяет, что забытые факты не восстанавливаются из старой переписки."""
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Старый секрет 4829")
    await client.delete("/api/v1/emo/memory", headers=auth["anna"])
    # Ten messages cause a summary refresh. The forgotten source must stay excluded.
    for _ in range(10):
        await send(client, auth, cid, "Новое сообщение")
    state = (await client.get("/api/v1/emo/state", headers=auth["anna"])).json()
    assert "4829" not in state["summary"]
    assert all("4829" not in f["text"] for f in state["facts"])

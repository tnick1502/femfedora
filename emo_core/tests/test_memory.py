from tests.conftest import conversation, send


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

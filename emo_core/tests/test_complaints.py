import pytest

from tests.conftest import conversation, send


async def test_recurrence_and_history(client, auth):
    """Проверяет объединение повторных упоминаний жалобы и происхождение наблюдений."""
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно")
    await send(client, auth, cid, "Я снова тревожусь и устала")
    rows = (await client.get("/api/v1/emo/complaints", headers=auth["anna"])).json()["items"]
    assert len(rows) == 2
    anxiety = next(c for c in rows if c["category"] == "anxiety")
    assert anxiety["metadata"]["observations"] == 2
    assert anxiety["user_confirmed"] is False
    await send(client, auth, cid, "Спасибо")
    current = (await client.get("/api/v1/emo/complaints", headers=auth["anna"])).json()["items"]
    assert all(c["status"] != "resolved" for c in current)


@pytest.mark.parametrize(
    "text", ["Я не тревожусь", "Подруга сказала: «Я тревожусь»", "Раньше я тревожилась"]
)
async def test_negation_quote_past(client, auth, text):
    """Проверяет отсутствие новых жалоб из отрицаний, цитат и давних событий."""
    cid = await conversation(client, auth)
    await send(client, auth, cid, text)
    assert (await client.get("/api/v1/emo/complaints", headers=auth["anna"])).json()["items"] == []

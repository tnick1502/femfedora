"""Журнал: что записано о пользователе. Без текста сообщений и без чужих данных."""

from tests.conftest import conversation, send

P = "/api/v1/emo"


async def test_journal_merges_checkins_signals_and_safety(client, auth):
    """Проверяет объединение самооценок, сигналов и safety-событий в журнале пользователя."""
    await client.post(
        f"{P}/checkins",
        headers=auth["anna"],
        json={"mood_score": 4, "anxiety_score": 7, "sleep_quality": None},
    )
    cid = await conversation(client, auth)
    assert (await send(client, auth, cid, "Мне тревожно и одиноко")).status_code == 200
    assert (await send(client, auth, cid, "Я хочу убить себя сейчас")).status_code == 200

    data = (await client.get(f"{P}/journal", headers=auth["anna"])).json()
    kinds = [item["kind"] for item in data["items"]]
    assert {"checkin", "signals", "safety"} <= set(kinds)
    assert data["counts"]["checkin"] == 1 and data["not_a_diagnosis"] is True
    assert [item["at"] for item in data["items"]] == sorted(
        (item["at"] for item in data["items"]), reverse=True
    )
    checkin = next(item for item in data["items"] if item["kind"] == "checkin")
    assert (checkin["mood"], checkin["anxiety"], checkin["sleep"]) == (4, 7, None)
    signals = next(item for item in data["items"] if item["kind"] == "signals")
    assert signals["signals"] and all("code" in s for s in signals["signals"])
    safety = next(item for item in data["items"] if item["kind"] == "safety")
    assert safety["route"] == "urgent"
    # Текст реплик в журнал не попадает.
    assert "убить" not in str(data) and "тревожно" not in str(data)


async def test_journal_is_per_user_and_requires_auth(client, auth):
    """Проверяет изоляцию журнала, обязательный JWT и диапазон периода запроса."""
    await client.post(
        f"{P}/checkins",
        headers=auth["anna"],
        json={"mood_score": 5, "anxiety_score": 5, "sleep_quality": 5},
    )
    assert (await client.get(f"{P}/journal", headers=auth["maria"])).json()["items"] == []
    assert (await client.get(f"{P}/journal")).status_code in (401, 403)
    assert (await client.get(f"{P}/journal?days=0", headers=auth["anna"])).status_code == 422

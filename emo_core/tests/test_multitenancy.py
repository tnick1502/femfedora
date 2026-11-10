import pytest

from tests.conftest import conversation, send

P = "/api/v1/emo"


async def test_isolation_and_delete(client, auth):
    """Проверяет изоляцию бесед, сообщений и памяти, а также удаление только своего профиля."""
    a = await conversation(client, auth)
    b = await conversation(client, auth, "maria")
    assert (await send(client, auth, a, "Мне одиноко")).status_code == 200
    assert (await send(client, auth, b, "Я устала", "maria")).status_code == 200
    assert (
        await client.get(f"{P}/conversations/{a}/messages", headers=auth["maria"])
    ).status_code == 404
    assert (await send(client, auth, a, "чужой текст", "maria")).status_code == 404
    assert len((await client.get(f"{P}/complaints", headers=auth["anna"])).json()["items"]) == 1
    await client.post(
        f"{P}/checkins", headers=auth["anna"], json={"mood_score": 5, "anxiety_score": 3}
    )
    assert (await client.get(f"{P}/checkins", headers=auth["maria"])).json()["items"] == []
    assert (
        await client.patch(f"{P}/settings", headers=auth["anna"], json={"user_id": "x"})
    ).status_code == 422
    assert (await client.delete(f"{P}/data", headers=auth["anna"])).status_code == 200
    assert (await client.get(f"{P}/conversations", headers=auth["anna"])).status_code == 410
    assert (
        len(
            (await client.get(f"{P}/conversations/{b}/messages", headers=auth["maria"])).json()[
                "items"
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "path",
    [
        "/conversations",
        "/state",
        "/complaints",
        "/trends",
        "/checkins",
        "/questionnaires",
        "/questionnaire-attempts",
        "/next-actions",
        "/settings",
    ],
)
async def test_auth_required(client, path):
    """Проверяет обязательность JWT для пользовательских маршрутов."""
    assert (await client.get(P + path)).status_code == 401


async def test_user_id_rejected(client, auth):
    """Проверяет отказ от user_id в теле запроса вместо доверия переданному профилю."""
    assert (
        await client.post(
            P + "/conversations",
            headers=auth["anna"],
            json={"user_id": "11111111-1111-4111-8111-111111111111"},
        )
    ).status_code == 422


async def test_attempt_isolation(client, auth, enabled):
    """Проверяет недоступность чужой попытки шкалы при чтении и изменении."""
    response = await client.post(
        P + "/questionnaire-attempts", headers=auth["anna"], json={"questionnaire_id": enabled}
    )
    aid = response.json()["id"]
    assert (await client.get(P + "/questionnaire-attempts", headers=auth["maria"])).json()[
        "items"
    ] == []
    assert (
        await client.post(
            P + "/questionnaire-attempts", headers=auth["maria"], json={"attempt_id": aid}
        )
    ).status_code == 404

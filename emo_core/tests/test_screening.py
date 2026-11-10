import pytest
from sqlalchemy import select

from emo_core.core.exceptions import InvalidMaterial
from emo_core.db.models import Questionnaire, SafetyEvent
from emo_core.services.screening_engine import score_answers, validate_material
from tests.conftest import synthetic_material

P = "/api/v1/emo"


def test_exact_scoring_and_reverse():
    """Проверяет сумму прямых и обратных ключей и safety-флаг до завершения."""
    q = Questionnaire(**synthetic_material("EPDS"))
    answers = dict.fromkeys((str(i) for i in range(10)), 0)
    assert score_answers(q, answers, True) == (3, [])
    answers["1"] = 3
    answers["9"] = 1
    assert score_answers(q, answers, True) == (1, ["9"])
    assert score_answers(q, {"9": 1}, False) == (None, ["9"])


@pytest.mark.parametrize("code,score", [("GAD2", 6), ("GAD7", 21), ("PHQ9", 27)])
def test_scale_maximum(code, score):
    """Проверяет максимальную сумму искусственной шкалы для каждого кода."""
    q = Questionnaire(**synthetic_material(code))
    count = len(q.definition["items"])
    assert score_answers(q, dict.fromkeys((str(i) for i in range(count)), 3), True)[0] == score


def test_unapproved_blocked():
    """Проверяет отказ от редакции без одобрения или совпадения с проверенным материалом."""
    data = synthetic_material()
    data["definition"]["clinical_review_status"] = "pending"
    with pytest.raises(InvalidMaterial):
        validate_material(data)


async def test_disabled_registry(client, auth, sessions):
    """Проверяет исключение отключённой редакции из каталога доступных шкал."""
    async with sessions() as s, s.begin():
        q = await s.scalar(select(Questionnaire).where(Questionnaire.code == "EPDS"))
        q.enabled = False
        qid = str(q.id)
    result = (await client.get(P + "/questionnaires", headers=auth["anna"])).json()
    assert len(result["items"]) == 3 and result["unavailable"][0]["code"] == "EPDS"
    assert (
        await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"questionnaire_id": qid, "complete": True},
        )
    ).status_code == 409


async def test_all_attempts_partial_excluded_gad_next_step(client, auth, enabled):
    """Проверяет историю всех попыток, исключение черновиков из динамики и следующий шаг GAD-2."""
    partial = await client.post(
        P + "/questionnaire-attempts",
        headers=auth["anna"],
        json={"questionnaire_id": enabled, "answers": {"0": 1}},
    )
    assert partial.status_code == 200, partial.text
    assert partial.json()["total_score"] is None
    for answers in ({"0": 1, "1": 2}, {"0": 2, "1": 3}):
        result = await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"questionnaire_id": enabled, "answers": answers, "complete": True},
        )
        assert "GAD-7" in result.json()["next_step"]
    history = (await client.get(P + "/questionnaire-attempts", headers=auth["anna"])).json()[
        "items"
    ]
    assert len(history) == 3
    trends = (await client.get(P + "/trends", headers=auth["anna"])).json()["questionnaires"][
        "GAD2"
    ]
    assert [p["score"] for p in trends] == [3, 5]
    assert trends[1]["delta"] == 2
    assert (
        await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"attempt_id": history[0]["id"], "complete": True},
        )
    ).status_code == 409


async def test_safety_item_before_completion(client, auth, sessions):
    """Проверяет выдачу помощи по safety-пункту до завершения искусственной шкалы."""
    async with sessions() as s, s.begin():
        q = Questionnaire(**synthetic_material("EPDS"))
        s.add(q)
        await s.flush()
        qid = str(q.id)
    result = await client.post(
        P + "/questionnaire-attempts",
        headers=auth["anna"],
        json={"questionnaire_id": qid, "answers": {"9": 1}},
    )
    assert result.status_code == 200, result.text
    assert result.json()["total_score"] is None and result.json()["safety_flags"] == ["9"]
    assert "специалист" in result.json()["safety_guidance"]
    async with sessions() as s:
        assert await s.scalar(
            select(SafetyEvent.id).where(SafetyEvent.risk_type == "questionnaire_safety_item")
        )


async def test_cooldown(client, auth, enabled):
    """Проверяет отсрочку повторного предложения шкалы после отказа."""
    assert any(
        a.get("code") == "GAD2"
        for a in (await client.get(P + "/next-actions", headers=auth["anna"])).json()["items"]
    )
    await client.post(
        P + "/assessment-schedule/GAD2", headers=auth["anna"], json={"action": "declined"}
    )
    assert not any(
        a.get("code") == "GAD2"
        for a in (await client.get(P + "/next-actions", headers=auth["anna"])).json()["items"]
    )
    # Voluntary completion remains available during a declined-offer cooldown.
    assert (
        await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"questionnaire_id": enabled, "answers": {"0": 0, "1": 0}, "complete": True},
        )
    ).status_code == 200

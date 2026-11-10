from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin, Questionnaire
from tests.conftest import synthetic_material

P = "/api/v1/emo"


async def test_checkin_delta_stale(client, auth, sessions):
    """Проверяет разности самооценок и отсутствие актуальной динамики по устаревшим данным."""
    for mood, anxiety, sleep in [(4, 6, 3), (6, 4, 5)]:
        assert (
            await client.post(
                P + "/checkins",
                headers=auth["anna"],
                json={"mood_score": mood, "anxiety_score": anxiety, "sleep_quality": sleep},
            )
        ).status_code == 201
    data = (await client.get(P + "/trends", headers=auth["anna"])).json()["checkins"]
    assert data["sufficient_data"] and data["deltas"] == {
        "mood_score": 2,
        "anxiety_score": -2,
        "sleep_quality": 2,
    }
    async with sessions() as s, s.begin():
        rows = list((await s.scalars(select(Checkin).order_by(Checkin.created_at))).all())
        for i, r in enumerate(rows):
            r.created_at = utcnow() - timedelta(days=30 - i)
    data = (await client.get(P + "/trends", headers=auth["anna"])).json()["checkins"]
    assert not data["sufficient_data"] and data["deltas"] == {}


async def test_incomparable_versions(client, auth, sessions):
    """Проверяет отсутствие разностей между результатами разных редакций одной шкалы."""
    for version in ["v1", "v2"]:
        async with sessions() as s, s.begin():
            q = Questionnaire(**synthetic_material(version=version))
            s.add(q)
            await s.flush()
            qid = str(q.id)
        await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"questionnaire_id": qid, "answers": {"0": 1, "1": 1}, "complete": True},
        )
    data = (await client.get(P + "/trends", headers=auth["anna"])).json()["questionnaires"]["GAD2"]
    assert all(p["delta"] is None for p in data)

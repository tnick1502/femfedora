from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin
from emo_core.services.retention import prune


async def test_retention_removes_expired_only(client, auth, sessions):
    """Проверяет удаление только просроченных записей и сохранение свежих данных."""
    for name in ("anna", "maria"):
        await client.post(
            "/api/v1/emo/checkins", headers=auth[name], json={"mood_score": 5, "anxiety_score": 2}
        )
    async with sessions() as s, s.begin():
        row = await s.scalar(select(Checkin).order_by(Checkin.created_at))
        row.created_at = utcnow() - timedelta(days=400)
    assert await prune(sessions, 365) == 1
    async with sessions() as s:
        rows = list((await s.scalars(select(Checkin))).all())
        assert len(rows) == 1

from emo_core.db.models import Checkin


class WellbeingMonitor:
    async def save(self, session, user_id, data):
        """Добавляет самооценку пользователя с source=user_report и выполняет flush без commit."""
        checkin = Checkin(user_id=user_id, source="user_report", **data.model_dump())
        session.add(checkin)
        await session.flush()
        return checkin

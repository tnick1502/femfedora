from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Complaint, MessageSignal
from emo_core.schemas.emotion import CATEGORIES


class ComplaintManager:
    async def observe(self, session, user_id, message, analysis, threshold, pregnancy_id=None):
        """Сохраняет маркеры выше threshold и обновляет карточки эмоциональных жалоб.

        Для маркеров CATEGORIES создаёт либо активирует жалобу и повышает счётчик
        наблюдений. При unavailable-анализе записей не создаёт. Интенсивность
        из числа упоминаний не выводит.
        """
        if not analysis.available:
            return
        for code, confidence in analysis.markers.items():
            if confidence < threshold:
                continue
            session.add(
                MessageSignal(
                    user_id=user_id,
                    message_id=message.id,
                    signal_code=code,
                    confidence=confidence,
                    source=analysis.source,
                    model_version=analysis.model_version,
                    details={"not_diagnostic": True},
                )
            )
            if code not in CATEGORIES:
                continue
            complaint = await session.scalar(
                select(Complaint).where(Complaint.user_id == user_id, Complaint.category == code)
            )
            if complaint is None:
                complaint = Complaint(
                    user_id=user_id,
                    pregnancy_id=pregnancy_id,
                    category=code,
                    details={"observations": 1},
                    user_confirmed=False,
                )
                session.add(complaint)
            else:
                complaint.last_seen_at = utcnow()
                complaint.status = "active"
                complaint.details = {
                    **complaint.details,
                    "observations": complaint.details.get("observations", 0) + 1,
                }
        await session.flush()

    async def recurrence(self, session, user_id, now, window_days=30):
        """Считает разные даты и длительность наблюдений каждого маркера за window_days.

        days — число календарных дат UTC, span_days — расстояние между первой
        и последней. Повторы в один день не увеличивают days.
        """
        rows = (
            await session.execute(
                select(MessageSignal.signal_code, MessageSignal.observed_at).where(
                    MessageSignal.user_id == user_id,
                    MessageSignal.observed_at >= now - timedelta(days=window_days),
                )
            )
        ).all()
        days = {}
        for code, observed_at in rows:
            days.setdefault(code, set()).add(observed_at.date())
        return {
            code: {"days": len(dates), "span_days": (max(dates) - min(dates)).days}
            for code, dates in days.items()
        }

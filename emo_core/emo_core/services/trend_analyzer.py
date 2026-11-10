from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin, Complaint, MessageSignal, Questionnaire, TestAttempt


class TrendAnalyzer:
    async def analyze(self, session, user_id, days=90):
        """Собирает динамику шкал и check-in за days дней, жалобы и сигналы переписки.

        Разности баллов считает только между одинаковыми кодом, версией и языком шкалы.
        Для разностей check-in нужны минимум две записи и последняя не старше 14 дней.
        Упоминания жалоб считает за 7 и 30 дней; частота не служит оценкой тяжести.
        """
        now = utcnow()
        since = now - timedelta(days=days)
        attempts = (
            await session.execute(
                select(TestAttempt, Questionnaire)
                .join(Questionnaire)
                .where(
                    TestAttempt.user_id == user_id,
                    TestAttempt.status == "completed",
                    TestAttempt.completed_at >= since,
                )
                .order_by(TestAttempt.completed_at, TestAttempt.id)
            )
        ).all()
        scales, previous = {}, {}
        for attempt, questionnaire in attempts:
            key = (questionnaire.code, questionnaire.version, questionnaire.language)
            scales.setdefault(questionnaire.code, []).append(
                {
                    "at": attempt.completed_at,
                    "version": questionnaire.version,
                    "score": attempt.total_score,
                    "delta": attempt.total_score - previous[key] if key in previous else None,
                    "safety_flags": attempt.safety_flags,
                }
            )
            previous[key] = attempt.total_score
        checkins = list(
            (
                await session.scalars(
                    select(Checkin)
                    .where(Checkin.user_id == user_id, Checkin.created_at >= since)
                    .order_by(Checkin.created_at, Checkin.id)
                )
            ).all()
        )
        sufficient = len(checkins) >= 2 and checkins[-1].created_at >= now - timedelta(days=14)
        deltas = {}
        if sufficient:
            for name in ("mood_score", "anxiety_score", "sleep_quality"):
                values = [getattr(c, name) for c in checkins if getattr(c, name) is not None]
                deltas[name] = values[-1] - values[-2] if len(values) >= 2 else None
        complaints = list(
            (await session.scalars(select(Complaint).where(Complaint.user_id == user_id))).all()
        )
        signals = list(
            (
                await session.scalars(
                    select(MessageSignal).where(
                        MessageSignal.user_id == user_id,
                        MessageSignal.observed_at >= now - timedelta(days=30),
                    )
                )
            ).all()
        )
        return {
            "questionnaires": scales,
            "checkins": {
                "points": [
                    {
                        "at": c.created_at,
                        "mood": c.mood_score,
                        "anxiety": c.anxiety_score,
                        "sleep": c.sleep_quality,
                    }
                    for c in checkins
                ],
                "sufficient_data": sufficient,
                "deltas": deltas,
            },
            "complaints": [
                {
                    "category": c.category,
                    "status": c.status,
                    "first_seen_at": c.first_seen_at,
                    "last_seen_at": c.last_seen_at,
                    "mentions_7d": sum(
                        s.signal_code == c.category and s.observed_at >= now - timedelta(days=7)
                        for s in signals
                    ),
                    "mentions_30d": sum(s.signal_code == c.category for s in signals),
                    "message_ids": [s.message_id for s in signals if s.signal_code == c.category],
                }
                for c in complaints
            ],
            "direct_changes": [
                {
                    "code": s.signal_code,
                    "message_id": s.message_id,
                    "at": s.observed_at,
                    "source": s.source,
                }
                for s in signals
                if s.signal_code in ("self_reported_improvement", "self_reported_worsening")
            ],
            "interpretation": "Самооценки и наблюдения текста не являются диагнозом. Частота упоминаний не равна выраженности симптомов.",
        }

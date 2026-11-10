from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.base import utcnow
from emo_core.db.models import Checkin, MessageSignal, Questionnaire, SafetyEvent, TestAttempt
from emo_core.db.repositories import tenant_read_session
from emo_core.services.trend_analyzer import TrendAnalyzer

router = APIRouter()
JOURNAL_KINDS = ("checkin", "signals", "questionnaire", "safety")


@router.get("/trends")
async def trends(
    days: int = Query(90, ge=1, le=3650),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает динамику самооценок, завершённых шкал и жалоб за days дней."""
    return await TrendAnalyzer().analyze(session, user_id, days)


@router.get("/journal")
async def journal(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(200, ge=1, le=500),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Собирает ленту самооценок, шкал, маркеров и событий безопасности владельца JWT.

    Возвращает до limit записей за days дней, начиная с новых; counts считает
    все найденные события до ограничения выдачи, truncated отмечает усечение.
    Маркеры одной реплики и одного источника объединяются. Тексты сообщений
    в ленту не включаются.
    """
    since = utcnow() - timedelta(days=days)
    entries = []

    checkins = await session.scalars(
        select(Checkin).where(Checkin.user_id == user_id, Checkin.created_at >= since)
    )
    for row in checkins:
        entries.append(
            {
                "kind": "checkin",
                "at": row.created_at,
                "mood": row.mood_score,
                "anxiety": row.anxiety_score,
                "sleep": row.sleep_quality,
                "source": row.source,
            }
        )

    # Маркеры одного сообщения — одна запись журнала.
    signals = await session.scalars(
        select(MessageSignal)
        .where(MessageSignal.user_id == user_id, MessageSignal.observed_at >= since)
        .order_by(MessageSignal.observed_at, MessageSignal.id)
    )
    by_message = {}
    for row in signals:
        entry = by_message.setdefault(
            (row.message_id, row.source),
            {
                "kind": "signals",
                "at": row.observed_at,
                "message_id": row.message_id,
                "source": row.source,
                "model_version": row.model_version,
                "signals": [],
            },
        )
        entry["signals"].append(
            {
                "code": row.signal_code,
                "confidence": row.confidence,
                **(
                    {k: row.details[k] for k in ("status", "intensity") if k in row.details}
                    if row.source == "user_report"
                    else {}
                ),
            }
        )
    for entry in by_message.values():
        entry["signals"].sort(key=lambda s: -(s["confidence"] or 0))
        entries.append(entry)

    attempts = await session.execute(
        select(TestAttempt, Questionnaire)
        .join(Questionnaire)
        .where(
            TestAttempt.user_id == user_id,
            TestAttempt.status == "completed",
            TestAttempt.completed_at >= since,
        )
    )
    for attempt, questionnaire in attempts.all():
        entries.append(
            {
                "kind": "questionnaire",
                "at": attempt.completed_at,
                "attempt_id": attempt.id,
                "code": questionnaire.code,
                "score": attempt.total_score,
                "maximum": (questionnaire.scoring_rules or {}).get("maximum"),
                "safety_flag": bool(attempt.safety_flags),
            }
        )

    events = await session.scalars(
        select(SafetyEvent).where(SafetyEvent.user_id == user_id, SafetyEvent.created_at >= since)
    )
    for row in events:
        entries.append(
            {
                "kind": "safety",
                "at": row.created_at,
                "message_id": row.message_id,
                "route": row.risk_route,
                "risk": row.risk_type,
                "action": row.action_taken,
            }
        )

    entries.sort(key=lambda e: e["at"], reverse=True)
    counts = {kind: sum(e["kind"] == kind for e in entries) for kind in JOURNAL_KINDS}
    return {
        "items": entries[:limit],
        "counts": counts,
        "days": days,
        "truncated": len(entries) > limit,
        "not_a_diagnosis": True,
    }

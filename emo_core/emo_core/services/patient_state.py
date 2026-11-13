"""Bounded, source-tagged patient context assembled by functions, never by a free-form LLM.

No combined mental-health score. Text confidence is not symptom severity. Historical
observations stay historical; silence is never interpreted as recovery.
"""

from datetime import timedelta

from sqlalchemy import func, select

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin, MessageSignal, UserSettings
from emo_core.schemas.emotion import CATEGORIES
from emo_core.services.complaint_manager import ComplaintManager
from emo_core.services.perinatal import perinatal_context, same_assessment_episode
from emo_core.services.policies import policy
from emo_core.services.screening_engine import recent_results
from emo_core.services.screening_scheduler import (
    SCALE_CODES,
    STAGE_SCALES,
    ScreeningScheduler,
    assessment_history,
)


def scale_history(history, now, user):
    """Latest two completions of each scale; compare only identical versions and episodes."""
    grouped = {}
    for attempt, questionnaire in history:
        if attempt.status == "completed":
            grouped.setdefault(questionnaire.code, []).append((attempt, questionnaire))
    result = []
    for code in SCALE_CODES:
        if code not in grouped:
            continue
        (last, q), *older = grouped[code]
        saved = last.result or {}
        item = {
            "code": code,
            "score": last.total_score,
            "maximum": saved.get("maximum_score"),
            "at": last.completed_at.date().isoformat(),
            "version": q.version,
            "band": (saved.get("interpretation") or {}).get("code"),
            "followup": saved.get("recommended_action")
            in ("immediate_safety_assessment", "professional_assessment"),
            "safety_flags": bool(last.safety_flags),
            "stale": last.completed_at < now - timedelta(days=30),
            "other_episode": not same_assessment_episode(last, user),
        }
        if older:
            previous, pq = older[0]
            if (
                (q.version, q.language, last.pregnancy_id, last.context.get("birth_date"))
                == (
                    pq.version,
                    pq.language,
                    previous.pregnancy_id,
                    previous.context.get("birth_date"),
                )
                and last.total_score is not None
                and previous.total_score is not None
            ):
                item.update(
                    previous=previous.total_score,
                    previous_at=previous.completed_at.date().isoformat(),
                    delta=last.total_score - previous.total_score,
                )
        result.append(item)
    return result


async def short_context(session, user_id, now, current_message_id=None, pregnancy_id=None):
    """Three-day marker memory, aggregated to at most 24 rows and never promoted to current fact."""
    rows = (
        await session.execute(
            select(
                MessageSignal.signal_code,
                func.max(MessageSignal.observed_at),
                func.count(func.distinct(MessageSignal.message_id)),
            )
            .where(
                MessageSignal.user_id == user_id,
                MessageSignal.observed_at >= now - timedelta(hours=72),
                MessageSignal.message_id != current_message_id if current_message_id else True,
                MessageSignal.details["pregnancy_id"].as_string() == str(pregnancy_id)
                if pregnancy_id
                else MessageSignal.details["pregnancy_id"].as_string().is_(None),
            )
            .group_by(MessageSignal.signal_code)
        )
    ).all()
    return [
        {"code": code, "last": at.isoformat(), "mentions": count, "source": "text_observation"}
        for code, at, count in rows
        if code in CATEGORIES
        or code
        in (
            "self_reported_improvement",
            "self_reported_worsening",
            "functional_impact",
            "possible_self_harm",
            "self_harm_intent",
            "possible_psychosis",
            "possible_harm_to_baby",
            "possible_violence_exposure",
            "maternal_estrangement",
            "possible_urgent_medical_issue",
        )
    ]


async def checkin_context(session, user_id, now):
    # Select the latest observation on each UTC day before limiting to two days.
    ranked = (
        select(
            Checkin.id,
            func.row_number()
            .over(
                partition_by=func.date(Checkin.created_at),
                order_by=(Checkin.created_at.desc(), Checkin.id.desc()),
            )
            .label("rank"),
        )
        .where(Checkin.user_id == user_id, Checkin.created_at <= now)
        .subquery()
    )
    rows = list(
        (
            await session.scalars(
                select(Checkin)
                .join(ranked, ranked.c.id == Checkin.id)
                .where(ranked.c.rank == 1)
                .order_by(Checkin.created_at.desc(), Checkin.id.desc())
                .limit(2)
            )
        ).all()
    )
    if not rows:
        return {}
    last = rows[0]
    result = {
        "at": last.created_at.date().isoformat(),
        "mood": last.mood_score,
        "anxiety": last.anxiety_score,
        "sleep": last.sleep_quality,
        "stale": last.created_at < now - timedelta(days=14),
    }
    # Same-day check-ins are not evidence of a trend over days.
    if (
        len(rows) == 2
        and not result["stale"]
        and rows[1].created_at >= now - timedelta(days=30)
        and rows[1].created_at.date() != last.created_at.date()
    ):
        result["previous_at"] = rows[1].created_at.date().isoformat()
        result["delta"] = {
            k: getattr(last, field) - getattr(rows[1], field)
            for k, field in (
                ("mood", "mood_score"),
                ("anxiety", "anxiety_score"),
                ("sleep", "sleep_quality"),
            )
            if getattr(last, field) is not None and getattr(rows[1], field) is not None
        }
    return result


class PatientStateBuilder:
    def __init__(self, settings):
        self.settings = settings

    async def build(
        self,
        session,
        user_id,
        analysis=None,
        perinatal=None,
        message_id=None,
        now=None,
        *,
        history=None,
        screening=None,
    ):
        now = now or utcnow()
        user = await session.get(UserSettings, user_id)
        perinatal = perinatal or perinatal_context(user, now)
        history = history if history is not None else await assessment_history(session, user_id)
        if screening is None:
            screening = await recent_results(
                session,
                user_id,
                now,
                policy(self.settings, "safety_rules.yaml").get("screening_result_valid_days", 30),
                history=history,
            )
        scales = scale_history(history, now, user)
        for scale in scales:
            scale["applicable"] = scale["code"] in STAGE_SCALES[perinatal.stage]
        current = {
            k: round(v, 2)
            for k, v in (analysis.markers.items() if analysis and analysis.available else [])
            if v >= self.settings.jev_review_threshold
        }
        short = {
            "scope": analysis.scope if analysis else "history_only",
            "current": current,
            "recent": await short_context(session, user_id, now, message_id, user.pregnancy_id),
        }
        conflicts = []
        if current.get("self_reported_worsening", 0) >= self.settings.jev_threshold:
            if any(not r["stale"] and not r["followup"] and not r["other_episode"] for r in scales):
                conflicts.append("current_worsening_despite_low_screen")
        if current.get("self_reported_improvement", 0) >= self.settings.jev_threshold and any(
            r["followup"] and not r["stale"] and not r["other_episode"] for r in scales
        ):
            conflicts.append("improvement_does_not_clear_screening_followup")
        return {
            "version": 2,
            "as_of": now.isoformat(timespec="seconds"),
            "short": short,
            "long": {
                "scales": scales,
                "screening_safety": [
                    {"code": r["code"], "at": r["safety_at"], "needs_assessment": True}
                    for r in screening
                    if r.get("safety_flags")
                ],
                "checkin": await checkin_context(session, user_id, now),
                "patterns": await ComplaintManager().recurrence(
                    session,
                    user_id,
                    now,
                    policy(self.settings, "safety_rules.yaml")
                    .get("persistence", {})
                    .get("window_days", 30),
                    user.pregnancy_id,
                ),
            },
            "assessments": await ScreeningScheduler(self.settings).plan(
                session, user_id, now, history=history
            ),
            "conflicts": conflicts,
        }


def prompt_patient_state(state):
    """Small loss-aware projection; full dated evidence remains in /state and the DB."""
    from copy import deepcopy

    result = deepcopy(state)
    result["as_of"] = result["as_of"][:16]
    for row in result["short"]["recent"]:
        row.pop("source", None)  # All entries here are explicitly historical text observations.
        row["last"] = row["last"][:16]
    result["assessments"] = [
        {
            k: v
            for k, v in row.items()
            if k in ("code", "status", "due_at", "remind_at", "interval_days") and v is not None
        }
        for row in result["assessments"]
    ]
    for row in result["assessments"]:
        for key in ("due_at", "remind_at"):
            if key in row:
                row[key] = row[key][:10]
    return result

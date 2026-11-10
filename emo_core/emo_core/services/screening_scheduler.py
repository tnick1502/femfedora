from datetime import timedelta

from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import (
    AssessmentSchedule,
    Checkin,
    Complaint,
    Questionnaire,
    SafetyEvent,
    TestAttempt,
    UserSettings,
)
from emo_core.services.perinatal import perinatal_context
from emo_core.services.policies import policy

ELEVATED_ACTIONS = ("immediate_safety_assessment", "professional_assessment")
GAD7_TRIGGER = "gad2_followup_or_professional"


class ScreeningScheduler:
    """Когда уместно предложить шкалу. Только чтение в next_actions; запись — в record()."""

    def __init__(self, settings):
        """Загружает интервалы шкал и срочного маршрута; проверяет допуск предложений в production."""
        self.rules = policy(settings, "screening_policy.yaml")
        self.urgent_cooldown_days = policy(settings, "safety_rules.yaml").get(
            "urgent_cooldown_days", 30
        )
        self.offers_approved = (
            settings.app_env != "production"
            or self.rules.get("clinical_review_status") == "approved"
        )

    def window(self, user, perinatal, now):
        """Возвращает название текущего перинатального окна и дату его начала либо пару None."""
        low, high = self.rules["postpartum_weeks"]
        if perinatal.stage == "postpartum" and low <= perinatal.postpartum_weeks <= high:
            return "postpartum", user.birth_date + timedelta(weeks=low)
        low, high = self.rules["late_pregnancy"]
        if perinatal.stage == "pregnant" and low <= perinatal.gestational_week <= high:
            return "late_pregnancy", now - timedelta(weeks=perinatal.gestational_week - low + 1)
        return None, None

    def due_reason(self, code, schedule, last, window, window_start, complaints, now):
        """Возвращает код основания предложить шкалу либо None.

        Учитывает первое прохождение, минимальный интервал, перинатальное окно,
        повышенный прошлый результат и жалобы. GAD-7 предлагает только по отдельному
        триггеру следующего шага после GAD-2.
        """
        if code == "GAD7":
            # Второй шаг после GAD-2 от 3 баллов, не самостоятельный первичный тест.
            due = (
                schedule is not None
                and schedule.trigger_reason == GAD7_TRIGGER
                and schedule.next_due_at is not None
                and schedule.next_due_at <= now
            )
            return GAD7_TRIGGER if due else None
        if last is None:
            return "initial" if self.rules.get("initial", True) else None
        completed_at = last.completed_at
        if completed_at > now - timedelta(days=self.rules["minimum_reoffer_days"]):
            return None
        if window and completed_at < window_start:
            return window
        elevated = (last.result or {}).get("recommended_action") in ELEVATED_ACTIONS or bool(
            last.safety_flags
        )
        if (
            code in ("EPDS", "PHQ9")
            and elevated
            and completed_at
            <= now - timedelta(days=self.rules.get("recheck_after_elevated_days", 14))
        ):
            return "recheck_after_elevated"
        if complaints and completed_at <= now - timedelta(
            days=self.rules.get("complaint_reoffer_days", 30)
        ):
            return "new_or_recurring_complaints"
        return None

    async def next_actions(self, session, user_id, now=None):
        """Возвращает доступные предложения шкал и недельного check-in без записи в БД.

        Учитывает редакции, черновики, отказы, сроки, жалобы и перинатальный этап.
        После недавнего urgent возвращает только professional_support. now задаёт
        момент проверки; по умолчанию используется текущее UTC-время.
        """
        now = now or utcnow()
        urgent = await session.scalar(
            select(SafetyEvent.id)
            .where(
                SafetyEvent.user_id == user_id,
                SafetyEvent.risk_route == "urgent",
                SafetyEvent.created_at >= now - timedelta(days=self.urgent_cooldown_days),
            )
            .limit(1)
        )
        if urgent:
            # Помощь не откладывается ради повторного теста (ТЗ §10.2).
            return [
                {"type": "professional_support", "reason": "safety_priority", "optional": False}
            ]
        user = await session.get(UserSettings, user_id)
        perinatal = perinatal_context(user, now)
        window, window_start = self.window(user, perinatal, now)
        enabled = list(
            (
                await session.scalars(select(Questionnaire).where(Questionnaire.enabled.is_(True)))
            ).all()
        )
        codes = {q.code for q in enabled} if self.offers_approved else set()
        if "EPDS" in codes and not self.rules["allow_phq9_with_epds"]:
            codes.discard("PHQ9")
        complaints = list(
            (
                await session.scalars(
                    select(Complaint).where(
                        Complaint.user_id == user_id,
                        Complaint.status.in_(["new", "active"]),
                        Complaint.last_seen_at >= now - timedelta(days=30),
                    )
                )
            ).all()
        )
        attempts = (
            await session.execute(
                select(TestAttempt, Questionnaire.code)
                .join(Questionnaire)
                .where(TestAttempt.user_id == user_id)
                .order_by(TestAttempt.completed_at.desc())
            )
        ).all()
        draft_codes = {code for attempt, code in attempts if attempt.status == "in_progress"}
        last_completed = {}
        for attempt, code in attempts:
            if attempt.status == "completed" and code not in last_completed:
                last_completed[code] = attempt
        schedules = {
            row.questionnaire_code: row
            for row in (
                await session.scalars(
                    select(AssessmentSchedule).where(AssessmentSchedule.user_id == user_id)
                )
            ).all()
        }
        result = []
        for code in sorted(codes):
            if code in draft_codes:
                continue
            schedule = schedules.get(code)
            if schedule is not None:
                if schedule.snoozed_until and schedule.snoozed_until > now:
                    continue
                if schedule.last_offered_at and schedule.last_offered_at > now - timedelta(
                    days=self.rules["minimum_reoffer_days"]
                ):
                    continue
                if code != "GAD7" and schedule.next_due_at and schedule.next_due_at > now:
                    continue
            reason = self.due_reason(
                code, schedule, last_completed.get(code), window, window_start, complaints, now
            )
            if reason is None:
                continue
            result.append(
                {
                    "type": "questionnaire",
                    "code": code,
                    "reason": reason,
                    "policy_version": self.rules["version"],
                    "optional": True,
                }
            )
        last = await session.scalar(
            select(Checkin)
            .where(Checkin.user_id == user_id)
            .order_by(Checkin.created_at.desc())
            .limit(1)
        )
        interval = timedelta(days=self.rules.get("checkin_interval_days", 7))
        if last is None or last.created_at <= now - interval:
            result.append({"type": "checkin", "optional": True, "reason": "weekly"})
        return result

    async def record(self, session, user_id, code, action):
        """Записывает показ, отказ или отсрочку шкалы и версию правил.

        Создаёт расписание при отсутствии. Показ обновляет срок повторного предложения,
        кроме GAD-7; остальные действия устанавливают snoozed_until. Commit не выполняет.
        """
        row = await session.scalar(
            select(AssessmentSchedule).where(
                AssessmentSchedule.user_id == user_id, AssessmentSchedule.questionnaire_code == code
            )
        )
        if row is None:
            row = AssessmentSchedule(user_id=user_id, questionnaire_code=code)
            session.add(row)
        now = utcnow()
        if action == "offered":
            row.last_offered_at = now
            if code != "GAD7":
                row.next_due_at = now + timedelta(days=self.rules["minimum_reoffer_days"])
        else:
            row.snoozed_until = now + timedelta(days=self.rules["decline_cooldown_days"])
        row.policy_version = self.rules["version"]

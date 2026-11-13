"""Stage-aware assessment plan. Intervals are product policy, not diagnostic standards."""

from datetime import timedelta

from sqlalchemy import String, case, func, select

from emo_core.db.base import utcnow
from emo_core.db.models import (
    AssessmentSchedule,
    Checkin,
    Questionnaire,
    SafetyEvent,
    TestAttempt,
    UserSettings,
)
from emo_core.services.perinatal import perinatal_context, same_assessment_episode
from emo_core.services.policies import policy

STAGE_SCALES = {
    "preconception": ("PHQ9", "GAD7"),
    "pregnant": ("PHQ9", "GAD7", "EPDS"),
    "postpartum": ("PHQ9", "GAD7", "EPDS", "CITYBITS"),
    "after_loss": ("PHQ9", "GAD7"),
    "unknown": ("PHQ9", "GAD7"),
}
SCALE_CODES = ("PHQ9", "GAD7", "EPDS", "CITYBITS")
SCALE_NAMES = {
    "PHQ9": "PHQ-9",
    "GAD7": "GAD-7",
    "EPDS": "EPDS",
    "CITYBITS": "City Birth Trauma Scale",
}
ELEVATED_ACTIONS = ("immediate_safety_assessment", "professional_assessment")


async def assessment_history(session, user_id):
    """One shared read: latest two completions, latest draft and latest safety observation (<=16 rows)."""
    ranked = (
        select(
            TestAttempt.id.label("id"),
            func.row_number()
            .over(
                partition_by=(Questionnaire.code, TestAttempt.status),
                order_by=(
                    func.coalesce(TestAttempt.completed_at, TestAttempt.started_at).desc(),
                    TestAttempt.id.desc(),
                ),
            )
            .label("rank"),
            func.row_number()
            .over(
                partition_by=Questionnaire.code,
                order_by=(
                    case((TestAttempt.safety_flags != [], 0), else_=1),
                    func.coalesce(
                        TestAttempt.context["safety_observed_at"].as_string(),
                        func.replace(TestAttempt.completed_at.cast(String), " ", "T"),
                        func.replace(TestAttempt.started_at.cast(String), " ", "T"),
                    ).desc(),
                    TestAttempt.id.desc(),
                ),
            )
            .label("safety_rank"),
        )
        .join(Questionnaire)
        .where(
            TestAttempt.user_id == user_id,
            Questionnaire.code.in_(SCALE_CODES),
            TestAttempt.status.in_(("completed", "in_progress")),
        )
        .subquery()
    )
    rows = (
        await session.execute(
            select(TestAttempt, Questionnaire)
            .join(Questionnaire)
            .join(ranked, ranked.c.id == TestAttempt.id)
            .where(
                (ranked.c.rank == 1)
                | ((TestAttempt.status == "completed") & (ranked.c.rank == 2))
                | ((ranked.c.safety_rank == 1) & (TestAttempt.safety_flags != []))
            )
            .order_by(
                func.coalesce(TestAttempt.completed_at, TestAttempt.started_at).desc(),
                TestAttempt.id.desc(),
            )
        )
    ).all()
    return rows


class ScreeningScheduler:
    def __init__(self, settings):
        self.rules = policy(settings, "screening_policy.yaml")
        self.urgent_cooldown_days = policy(settings, "safety_rules.yaml").get(
            "urgent_cooldown_days", 30
        )
        self.offers_approved = settings.app_env != "production" or all(
            (
                self.rules.get("clinical_review_status") == "approved",
                self.rules.get("reviewed_by"),
                self.rules.get("approval_reference"),
            )
        )

    def interval(self, code, result=None):
        rule = self.rules.get("scales", {}).get(code, {})
        elevated = result and (
            (result.result or {}).get("recommended_action") in ELEVATED_ACTIONS
            or result.safety_flags
        )
        return rule.get("elevated_repeat_days", 14) if elevated else rule.get("repeat_days", 30)

    def stage_start(self, user, perinatal, now):
        if perinatal.stage == "postpartum":
            return user.birth_date
        if perinatal.stage == "pregnant" and perinatal.gestational_week is not None:
            return now - timedelta(weeks=perinatal.gestational_week)
        return None

    async def plan(self, session, user_id, now=None, *, history=None):
        """Four bounded status rows, including missing forms, drafts, snoozes and due dates."""
        now = now or utcnow()
        user = await session.get(UserSettings, user_id)
        if user is None:
            return []
        perinatal = perinatal_context(user, now)
        allowed = STAGE_SCALES[perinatal.stage]
        registry = list(
            (
                await session.scalars(
                    select(Questionnaire).where(
                        Questionnaire.enabled.is_(True),
                        Questionnaire.code.in_(SCALE_CODES),
                    )
                )
            ).all()
        )
        enabled = {q.code for q in registry}
        history = history if history is not None else await assessment_history(session, user_id)
        attempts = {}
        for attempt, questionnaire in history:
            attempts.setdefault((questionnaire.code, attempt.status), attempt)
        schedules = {
            r.questionnaire_code: r
            for r in (
                await session.scalars(
                    select(AssessmentSchedule).where(AssessmentSchedule.user_id == user_id)
                )
            ).all()
        }
        start = self.stage_start(user, perinatal, now)
        rows = []
        for code in SCALE_CODES:
            last = attempts.get((code, "completed"))
            draft = attempts.get((code, "in_progress"))
            schedule = schedules.get(code)
            interval = self.interval(code, last)
            reason = "initial"
            due = now
            if last:
                due = last.completed_at + timedelta(days=interval)
                reason = (
                    "recheck_after_elevated"
                    if (
                        (last.result or {}).get("recommended_action") in ELEVATED_ACTIONS
                        or last.safety_flags
                    )
                    else "periodic"
                )
                # A new stage/episode needs a fresh baseline; minimum interval still applies.
                other_episode = not same_assessment_episode(last, user)
                changed_stage = last.context.get("perinatal_stage") not in (None, perinatal.stage)
                if other_episode or changed_stage or (start and last.completed_at < start):
                    due = last.completed_at + timedelta(days=self.rules["minimum_reoffer_days"])
                    reason = "stage_baseline"
                elif perinatal.stage == "pregnant" and perinatal.gestational_week is not None:
                    low, high = self.rules["late_pregnancy"]
                    window_start = now - timedelta(weeks=perinatal.gestational_week - low)
                    if (
                        low <= perinatal.gestational_week <= high
                        and last.completed_at < window_start
                    ):
                        due = last.completed_at + timedelta(days=self.rules["minimum_reoffer_days"])
                        reason = "late_pregnancy"
            first_allowed = None
            if code == "CITYBITS" and user.birth_date:
                first_allowed = user.birth_date + timedelta(
                    days=self.rules.get("citybits_first_after_birth_days", 42)
                )
                due = max(due if last else first_allowed, first_allowed)
            row = {
                "code": code,
                "name": SCALE_NAMES[code],
                "stage": perinatal.stage,
                "interval_days": interval,
                "last_completed_at": last.completed_at.isoformat() if last else None,
                "last_offered_at": schedule.last_offered_at.isoformat()
                if schedule and schedule.last_offered_at
                else None,
                "due_at": due.isoformat(),
                "reason": reason,
                "status": "due" if due <= now else "up_to_date",
            }
            if code not in allowed:
                row.update(status="not_applicable", reason="stage", due_at=None)
            elif code not in enabled:
                row.update(
                    status="unavailable",
                    reason="Редакция отключена или отсутствует",
                )
            elif not self.offers_approved:
                row.update(status="unavailable", reason="screening_policy_not_approved")
            elif code == "CITYBITS" and not user.birth_date:
                row.update(status="waiting_window", reason="birth_date_required", due_at=None)
            elif draft:
                # Resume only drafts of this episode/stage (old drafts stay readable in history).
                same_episode = same_assessment_episode(draft, user)
                same_stage = draft.context.get("perinatal_stage") == perinatal.stage
                newer_completion = last and last.completed_at >= draft.started_at
                if same_episode and same_stage and not newer_completion:
                    row.update(status="in_progress", attempt_id=str(draft.id), reason="resume")
            if (
                row["status"] in ("due", "in_progress", "up_to_date")
                and first_allowed
                and first_allowed > now
            ):
                row.update(status="waiting_window", reason="postpartum_window")
            if row["status"] in ("due", "in_progress"):
                if schedule and schedule.snoozed_until and schedule.snoozed_until > now:
                    row.update(status="deferred", remind_at=schedule.snoozed_until.isoformat())
                elif (
                    schedule
                    and schedule.last_offered_at
                    and schedule.last_offered_at
                    > now - timedelta(days=self.rules["minimum_reoffer_days"])
                ):
                    row.update(
                        status="offered_recently",
                        remind_at=(
                            schedule.last_offered_at
                            + timedelta(days=self.rules["minimum_reoffer_days"])
                        ).isoformat(),
                    )
            rows.append(row)
        return rows

    async def next_actions(self, session, user_id, now=None, plan=None, *, screening=None):
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
            return [
                {"type": "professional_support", "reason": "safety_priority", "optional": False}
            ]
        if screening is None:
            # Local import avoids the scoring engine / scheduler module dependency cycle.
            from emo_core.services.screening_engine import recent_results

            screening = await recent_results(session, user_id, now)
        if any(r.get("safety_flags") for r in screening):
            return [
                {
                    "type": "professional_support",
                    "reason": "screening_safety_priority",
                    "optional": False,
                }
            ]
        rows = plan if plan is not None else await self.plan(session, user_id, now)
        result = [
            {
                **row,
                "type": "questionnaire",
                "optional": True,
                "policy_version": self.rules["version"],
            }
            for row in rows
            if row["status"] in ("due", "in_progress")
        ]
        # Prefer a started form, then the oldest due date; stable core-scale ordering for ties.
        result.sort(key=lambda row: (row["status"] != "in_progress", row["due_at"]))
        last = await session.scalar(
            select(Checkin)
            .where(Checkin.user_id == user_id)
            .order_by(Checkin.created_at.desc())
            .limit(1)
        )
        if last is None or last.created_at <= now - timedelta(
            days=self.rules.get("checkin_interval_days", 7)
        ):
            result.append({"type": "checkin", "optional": True, "reason": "weekly"})
        return result

    async def record(self, session, user_id, code, action):
        row = await session.scalar(
            select(AssessmentSchedule).where(
                AssessmentSchedule.user_id == user_id,
                AssessmentSchedule.questionnaire_code == code,
            )
        )
        if row is None:
            row = AssessmentSchedule(user_id=user_id, questionnaire_code=code)
            session.add(row)
        now = utcnow()
        if action == "offered":
            row.last_offered_at = now
        else:
            row.snoozed_until = now + timedelta(days=self.rules["decline_cooldown_days"])
        row.policy_version = self.rules["version"]

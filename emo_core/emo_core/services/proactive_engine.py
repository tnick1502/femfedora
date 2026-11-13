from datetime import time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from emo_core.db.base import utcnow
from emo_core.db.models import (
    Conversation,
    Message,
    NotificationOutbox,
    ProactiveJob,
    SafetyEvent,
    UserSettings,
)
from emo_core.services.policies import policy
from emo_core.services.screening_scheduler import SCALE_NAMES, ScreeningScheduler

TEMPLATES = {
    "weekly_checkin": "Привет! Как ты себя чувствуешь в последние дни? Если хочется, можем немного поговорить.",
    "questionnaire_due": "Если тебе удобно, можно пройти короткий опрос о самочувствии. Это необязательно; доступные опросники есть в приложении.",
    "agreed_followup": "Мы договорились вернуться к разговору. Как ты сегодня себя чувствуешь? Можно ответить, когда будет удобно.",
}
PRIORITY = {"agreed_followup": 0, "questionnaire_due": 1, "weekly_checkin": 2}


def within(value, start, end):
    """Проверяет попадание времени в полуоткрытый интервал [start, end).

    Поддерживает переход через полночь; совпадающие границы означают весь день.
    """
    if start == end:
        return True
    return start <= value < end if start < end else value >= start or value < end


class ProactiveMessagingEngine:
    def __init__(self, settings, sessions):
        """Загружает правила инициативных сообщений, их версию и паузу после срочного маршрута."""
        self.settings, self.sessions = settings, sessions
        document = policy(settings, "proactive_policy.yaml")
        self.rules, self.policy_version = document["proactive"], str(document.get("version", "1.0"))
        self.urgent_cooldown_days = policy(settings, "safety_rules.yaml").get(
            "urgent_cooldown_days", 30
        )

    def allowed_time(self, user, now):
        """Проверяет разрешённое окно и тихие часы в часовом поясе пользователя.

        now переводит в IANA-пояс профиля. Некорректный пояс или время дают False.
        """
        try:
            local = now.astimezone(ZoneInfo(user.timezone)).time().replace(tzinfo=None)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            return False
        return within(
            local,
            time.fromisoformat(self.rules["allowed_start"]),
            time.fromisoformat(self.rules["allowed_end"]),
        ) and not within(local, user.quiet_hours_start, user.quiet_hours_end)

    async def enqueue(self, session, **values):
        """Добавляет задание, игнорируя повтор существующего deduplication_key."""
        insert = pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert
        await session.execute(
            insert(ProactiveJob)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["deduplication_key"])
        )

    async def plan(self, session, user, now):
        """Создаёт недостающие задания для последней активной беседы пользователя.

        Планирует недельный контакт и доступные напоминания шкал с ключами календарного
        периода. Для отключённой инициативности, деактивации или отсутствия беседы
        ничего не создаёт; ограничения фактической отправки проверяются отдельно.
        """
        if not user.proactive_enabled or user.deleted_at:
            return
        conversation = await session.scalar(
            select(Conversation)
            .where(Conversation.user_id == user.user_id, Conversation.status == "active")
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        if not conversation:
            return
        # Calendar buckets deduplicate discovery, eligibility also checks a rolling 7-day gap.
        bucket = now.date().toordinal() // self.rules["weekly_checkin_days"]
        await self.enqueue(
            session,
            user_id=user.user_id,
            conversation_id=conversation.id,
            job_type="weekly_checkin",
            deduplication_key=f"weekly:{user.user_id}:{bucket}",
            scheduled_for=now,
        )
        if self.rules["allow_questionnaire_reminders"]:
            actions = await ScreeningScheduler(self.settings).next_actions(
                session, user.user_id, now
            )
            for action in actions:
                if action["type"] == "questionnaire":
                    await self.enqueue(
                        session,
                        user_id=user.user_id,
                        conversation_id=conversation.id,
                        job_type="questionnaire_due",
                        deduplication_key=f"questionnaire:{user.user_id}:{action['code']}:{bucket}",
                        scheduled_for=now,
                        details={"code": action["code"]},
                    )

    async def eligibility(self, session, user, job, now):
        """Возвращает код препятствия отправке либо None, если контакт допустим.

        Проверяет согласие, паузы, время, срочный маршрут, доступность последнего
        анализа, активность, доставку и чтение, лимиты частоты и актуальность шкалы.
        При достигнутом лимите контактов без ответа записывает автоматическую паузу
        в настройки пользователя; остальные причины возвращает без отправки.
        """
        if not user.proactive_enabled or user.deleted_at:
            return "consent_disabled"
        if user.proactive_paused_until and user.proactive_paused_until > now:
            return "paused"
        if not self.allowed_time(user, now):
            return "outside_allowed_time"
        if job.job_type == "agreed_followup" and (
            not self.rules["allow_agreed_followups"] or not job.details.get("explicit_consent")
        ):
            return "followup_not_agreed"
        if job.job_type == "questionnaire_due" and not self.rules["allow_questionnaire_reminders"]:
            return "questionnaire_disabled"
        urgent = await session.scalar(
            select(SafetyEvent.id)
            .where(
                SafetyEvent.user_id == user.user_id,
                SafetyEvent.risk_route == "urgent",
                SafetyEvent.created_at >= now - timedelta(days=self.urgent_cooldown_days),
            )
            .limit(1)
        )
        if urgent:
            # Обычный worker не ведёт кризисное наблюдение (ТЗ §14.4).
            return "recent_urgent_route"
        latest_response = await session.scalar(
            select(Message)
            .where(
                Message.user_id == user.user_id,
                Message.role == "assistant",
                Message.origin != "proactive",
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        if latest_response and latest_response.details.get("analysis_available") is False:
            return "limited_safety_state"
        last_input = await session.scalar(
            select(Message.created_at)
            .where(Message.user_id == user.user_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        delivered = list(
            (
                await session.scalars(
                    select(Message)
                    .where(Message.user_id == user.user_id, Message.origin == "proactive")
                    .order_by(Message.created_at.desc())
                )
            ).all()
        )
        unanswered = [m for m in delivered if last_input is None or m.created_at > last_input]
        if len(unanswered) >= self.rules["unanswered_limit"]:
            user.proactive_paused_until = now + timedelta(days=self.rules["unanswered_pause_days"])
            user.proactive_pause_reason = "unanswered"
            return "unanswered_pause"
        if any(m.delivery_status != "delivered" for m in delivered):
            return "delivery_uncertain"
        if any(m.read_at is None for m in delivered):
            return "unread_proactive"
        if last_input and last_input > now - timedelta(
            hours=self.rules["suppress_if_user_active_within_hours"]
        ):
            return "recent_activity"
        if delivered and delivered[0].created_at > now - timedelta(
            hours=self.rules["minimum_gap_between_messages_hours"]
        ):
            return "minimum_gap"
        for days, maximum in (
            (7, self.rules["maximum_messages_7_days"]),
            (30, self.rules["maximum_messages_30_days"]),
        ):
            if sum(m.created_at > now - timedelta(days=days) for m in delivered) >= maximum:
                return "rate_limit"
        if job.job_type == "weekly_checkin" and any(
            m.created_at > now - timedelta(days=self.rules["weekly_checkin_days"])
            for m in delivered
        ):
            return "weekly_gap"
        if job.job_type == "questionnaire_due":
            actions = await ScreeningScheduler(self.settings).next_actions(
                session, user.user_id, now
            )
            if not any(a.get("code") == job.details.get("code") for a in actions):
                return "questionnaire_not_due"
        return None

    async def run_once(self, now=None, only_user=None):
        """Планирует и отправляет не более одного инициативного сообщения на пользователя.

        now позволяет проверить заданное время, only_user ограничивает цикл профилем.
        Блокирует профиль, затем задания через SKIP LOCKED; приоритет — согласованный
        контакт, шкала, недельный check-in. Сообщение, outbox и завершение задания
        фиксирует вместе. Возвращает счётчики sent/processed; модели не вызывает.
        """
        if not self.settings.proactive_worker_enabled:
            return {"sent": 0, "processed": 0}
        now = now or utcnow()
        async with self.sessions() as session:
            query = select(UserSettings.user_id).where(
                UserSettings.deleted_at.is_(None), UserSettings.proactive_enabled.is_(True)
            )
            if only_user is not None:
                query = query.where(UserSettings.user_id == only_user)
            ids = list((await session.scalars(query.order_by(UserSettings.user_id))).all())
        sent = processed = 0
        for user_id in ids:
            async with self.sessions() as session, session.begin():
                # Consistent ordering: user first, jobs second. API uses the same user lock.
                user = await session.scalar(
                    select(UserSettings)
                    .where(UserSettings.user_id == user_id)
                    .with_for_update(skip_locked=True)
                )
                if user is None:
                    continue
                await self.plan(session, user, now)
                jobs = list(
                    (
                        await session.scalars(
                            select(ProactiveJob)
                            .where(
                                ProactiveJob.user_id == user_id,
                                ProactiveJob.status == "pending",
                                ProactiveJob.scheduled_for <= now,
                            )
                            .with_for_update(skip_locked=True)
                        )
                    ).all()
                )
                jobs.sort(key=lambda j: (PRIORITY.get(j.job_type, 99), j.scheduled_for))
                for job in jobs:
                    reason = await self.eligibility(session, user, job, now)
                    if reason:
                        job.last_error = reason
                        if reason in (
                            "followup_not_agreed",
                            "questionnaire_disabled",
                            "questionnaire_not_due",
                            "consent_disabled",
                        ):
                            job.status = "cancelled"
                        else:
                            job.scheduled_for = now + timedelta(minutes=30)
                        continue
                    if job.job_type not in TEMPLATES:
                        job.status, job.last_error = "cancelled", "unsupported_job_type"
                        continue
                    content = TEMPLATES[job.job_type]
                    if job.job_type == "questionnaire_due":
                        content = (
                            f"Если тебе удобно, можно пройти или продолжить опросник "
                            f"{SCALE_NAMES[job.details['code']]} в разделе «Опросники». "
                            "Это добровольная самооценка, не диагноз; её можно отложить."
                        )
                    message = Message(
                        user_id=user_id,
                        conversation_id=job.conversation_id,
                        role="assistant",
                        origin="proactive",
                        content=content,
                        created_at=now,
                        details={
                            "job_type": job.job_type,
                            "proactive_policy_version": self.policy_version,
                        },
                    )
                    session.add(message)
                    await session.flush()
                    job.status, job.message_id, job.processed_at = "completed", message.id, now
                    job.attempt_count += 1
                    if user.push_enabled:
                        session.add(NotificationOutbox(user_id=user_id, message_id=message.id))
                    if job.job_type == "questionnaire_due":
                        await ScreeningScheduler(self.settings).record(
                            session, user_id, job.details["code"], "offered"
                        )
                    for other in jobs:
                        if other.id != job.id and other.status == "pending":
                            other.status, other.last_error = "superseded", "higher_priority_message"
                    sent += 1
                    processed += 1
                    break
        return {"sent": sent, "processed": processed}

from datetime import timedelta

from sqlalchemy import func, select

from emo_core.db.base import utcnow
from emo_core.db.models import Complaint, Message, UserSettings, UserState
from emo_core.services.perinatal import perinatal_context

# Верхняя граница числа реплик, загружаемых для сборки контекста одной беседы.
HISTORY_MESSAGES_LOADED = 3000
STALE_COMPLAINT_DAYS = 30


class MemoryManager:
    async def get(self, session, user_id):
        """Возвращает память профиля; при отсутствии создаёт пустую запись и выполняет flush."""
        state = await session.get(UserState, user_id)
        if state is None:
            state = UserState(
                user_id=user_id, summary="", preferences={}, recurring_topics=[], facts=[]
            )
            session.add(state)
            await session.flush()
        return state

    async def update(self, session, user_id, message, analysis, threshold=0.7):
        """Обновляет выдержки памяти и темы из пользовательских сообщений и анализа.

        Резюме обновляет при первом сообщении, каждом десятом или маркере изменения
        самочувствия. Берёт до 12 последних реплик после границы очистки, пропуская
        urgent в выдержках; сохраняет до 25 тем как предположения. LLM не вызывает.
        """
        state = await self.get(session, user_id)
        count = await session.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.user_id == user_id, Message.role == "user")
        )
        significant = any(
            analysis.markers.get(k, 0) >= threshold
            for k in ("self_reported_improvement", "self_reported_worsening")
        )
        if state.last_summarized_message_id is None or count % 10 == 0 or significant:
            rows = list(
                (
                    await session.scalars(
                        select(Message)
                        .where(
                            Message.user_id == user_id,
                            Message.role == "user",
                            Message.created_at > state.memory_cleared_at
                            if state.memory_cleared_at
                            else True,
                        )
                        .order_by(Message.created_at.desc(), Message.id.desc())
                        .limit(12)
                    )
                ).all()
            )
            corrected = [f for f in state.facts if f.get("source") == "user_edit"]
            state.facts = corrected + [
                {
                    "source": "user_report",
                    "message_id": str(m.id),
                    "observed_at": m.created_at.isoformat(),
                    "text": m.content[:300],
                }
                for m in reversed(rows)
                # Реплики, на которые был дан срочный ответ, не пересказываются в памяти:
                # иначе кризисная фраза возвращалась бы в контекст каждой следующей беседы.
                if (m.details or {}).get("route") != "urgent"
            ]
            # Explicit corrections survive automatic refresh; deletion still removes them.
            pinned = "\n".join(f["text"] for f in corrected)
            excerpts = "\n".join(
                f"{f['observed_at'][:10]}: {f['text']}"
                for f in state.facts
                if f.get("source") != "user_edit"
            )
            room = max(0, 3000 - len(pinned) - (1 if pinned else 0))
            state.summary = "\n".join(
                part for part in (pinned, excerpts[-room:] if room else "") if part
            )
            state.last_summarized_message_id = message.id
            state.updated_at = utcnow()
        topics = {
            x["category"]: x
            for x in state.recurring_topics
            if x.get("observed_at")
            and x["observed_at"] >= (utcnow() - timedelta(days=30)).isoformat()
        }
        for key, value in analysis.markers.items():
            if value >= threshold:
                topics[key] = {
                    "category": key,
                    "source": analysis.source,
                    "observed_at": utcnow().isoformat(),
                    "assumption": True,
                }
        state.recurring_topics = list(topics.values())[-25:]

    async def context(self, session, user_id, perinatal=None):
        """Собирает персональный контекст и возвращает пару (данные, жалобы).

        В данные входят память и перинатальный этап. Жалобы старше 30 дней отмечает
        как возможно устаревшие. Шкалы и самооценки собирает PatientStateBuilder один раз.
        """
        state = await self.get(session, user_id)
        now = utcnow()
        complaints = list(
            (
                await session.scalars(
                    select(Complaint).where(
                        Complaint.user_id == user_id, Complaint.status != "resolved"
                    )
                )
            ).all()
        )
        settings = await session.get(UserSettings, user_id)
        if perinatal is None:
            perinatal = perinatal_context(settings, now)
        data = {
            "summary": state.summary,
            "summary_note": "Выдержки из её прошлых реплик, в том числе из других бесед, с датами. Это фон, а не текущее сообщение: не приписывай их сегодняшнему дню и не пересказывай без повода.",
            "preferences": state.preferences,
            "memory_warning": "Сведения пользователя могут устареть; модельные маркеры — предположения, не диагнозы.",
            "complaints": [
                {
                    "category": c.category,
                    "status": c.status,
                    "last_seen_at": c.last_seen_at.isoformat(),
                    "confirmed": c.user_confirmed,
                    "other_episode": c.pregnancy_id != settings.pregnancy_id,
                    # Давние жалобы не считаются актуальными без уточнения (ТЗ §15).
                    "possibly_outdated": c.last_seen_at
                    < now - timedelta(days=STALE_COMPLAINT_DAYS),
                }
                for c in complaints
            ],
            "screenings_note": "Суммы шкал — самооценка, не диагноз. Не называй баллы признаком болезни и не пересчитывай их.",
            "perinatal": {
                "stage": perinatal.stage,
                "gestational_week": perinatal.gestational_week,
                "postpartum_weeks": perinatal.postpartum_weeks,
            },
        }
        return data, complaints

    async def history(self, session, user_id, conversation_id, after=None):
        """Возвращает до 3000 последних реплик собственной беседы в хронологическом порядке.

        Если задан after, читает только сообщения строго после этой даты.
        Возвращает текст, роль, дату ISO и признак шаблона безопасности.
        """
        query = select(Message).where(
            Message.user_id == user_id, Message.conversation_id == conversation_id
        )
        if after is not None:
            query = query.where(Message.created_at > after)
        rows = list(
            (
                await session.scalars(
                    query.order_by(Message.created_at.desc(), Message.id.desc()).limit(
                        HISTORY_MESSAGES_LOADED
                    )
                )
            ).all()
        )
        return [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "at": m.created_at.isoformat(),
                "template": m.origin == "safety_template",
            }
            for m in reversed(rows)
        ]

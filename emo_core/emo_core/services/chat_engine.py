"""Конвейер ответа в чате.

Обращения к внешним моделям выполняются вне транзакций базы данных. Запрос проходит три
коротких транзакции, каждая под блокировкой строки пользователя:

1. приём: проверка владения и идемпотентности, сохранение реплики со статусом «в обработке»;
2. решение: наблюдения, контекст, оценка состояния, выбор маршрута и протокола;
3. завершение: сохранение ответа, событий безопасности и памяти.

Между ними соединение с базой не удерживается, поэтому медленная модель не исчерпывает пул
и не блокирует другие запросы пользователя.

Тексты для внешних моделей проходят через границу приватности (обезличивание).
Контекст LLM дополнительно ограничивается бюджетом окна, сжатием и резюме ранней
части беседы. Повтор с тем же client_message_id дожидается результата текущей
обработки либо продолжает прерванную.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from time import perf_counter
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import select

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.db.base import utcnow
from emo_core.db.models import AssessmentSchedule, Message, SafetyEvent
from emo_core.db.repositories import ensure_user, owned_conversation, public
from emo_core.providers.mock_providers import DisabledJevProvider
from emo_core.schemas.emotion import CATEGORIES, EmotionAnalysis
from emo_core.services.anonymizer import boundary_for
from emo_core.services.complaint_manager import ComplaintManager
from emo_core.services.context_manager import ChatContextManager, ContextBudget, summary_request
from emo_core.services.decision_report import DecisionReport, describe_models
from emo_core.services.emotion_analyzer import EmotionAnalyzer
from emo_core.services.memory_manager import MemoryManager
from emo_core.services.patient_state import PatientStateBuilder, prompt_patient_state
from emo_core.services.perinatal import perinatal_context
from emo_core.services.safety_guard import SafetyGuard
from emo_core.services.safety_lexicon import LEXICON_VERSION
from emo_core.services.screening_engine import recent_results
from emo_core.services.screening_scheduler import (
    SCALE_NAMES,
    ScreeningScheduler,
    assessment_history,
)
from emo_core.services.state_evaluator import StateEvaluator
from emo_core.services.support_protocol_selector import SupportProtocolSelector

log = logging.getLogger(__name__)
LEARNED_NAMES_KEPT = 100
POLL_SECONDS = 0.25


class ChatEngine:
    def __init__(self, settings, llm, jev, models=None):
        """Сохраняет провайдеров и снимок моделей для обработки одной реплики.

        Время ожидания незавершённой обработки рассчитывает из тайм-аутов двух
        генераций, четырёх обращений Jev с повторами и десятисекундного запаса.
        """
        self.settings, self.llm, self.jev = settings, llm, jev
        self.models = models or describe_models(settings)
        # Отбор, полный анализ и две выходные проверки; SDK может повторить запрос.
        self.processing_timeout = timedelta(
            seconds=2 * settings.llm_timeout_seconds
            + 4 * settings.jev_timeout_seconds * (settings.jev_max_retries + 1)
            + 10
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tx(sessions):
        """Создаёт контекст короткой транзакции; блокировку владельца берёт вызывающий код."""
        return _UserTransaction(sessions)

    async def last_referral(self, session, user_id, since):
        """Находит время последней рекомендации специалиста среди недавних ответов.

        Просматривает до 50 последних ответов пользователя после since и ищет
        сохранённый referral_made. Если отметок нет, возвращает None.
        """
        rows = await session.scalars(
            select(Message)
            .where(
                Message.user_id == user_id,
                Message.role == "assistant",
                Message.created_at >= since,
            )
            .order_by(Message.created_at.desc())
            .limit(50)
        )
        return next((m.created_at for m in rows if m.details.get("referral_made")), None)

    @staticmethod
    async def _reply_to(session, user_id, conversation_id, message_id):
        """Находит ответ ассистента по reply_to в указанной собственной беседе либо None."""
        return await session.scalar(
            select(Message).where(
                Message.user_id == user_id,
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
                Message.details["reply_to"].as_string() == str(message_id),
            )
        )

    @staticmethod
    def _replay(message, assistant):
        """Собирает ответ API из сохранённой пары сообщений с признаком replayed=True."""
        return {
            "user_message": public(message),
            "assistant_message": public(assistant),
            "replayed": True,
        }

    def _fresh(self, message, now):
        """Проверяет, что обработка помечена pending и её время ещё не вышло за тайм-аут."""
        started = message.details.get("processing_started_at")
        return (
            message.details.get("processing") == "pending"
            and started is not None
            and datetime.fromisoformat(started) > now - self.processing_timeout
        )

    # ------------------------------------------------------------------ phase 1
    async def _accept(self, sessions, user_id, conversation_id, body):
        """Проверяет владельца и ключ повтора, затем сохраняет пользовательскую реплику.

        Возвращает пару (status, payload): new со снимком настроек, replay с готовой
        парой сообщений или wait с идентификатором обрабатываемой реплики.
        Прерванную обработку возобновляет; другой текст или беседа с прежним ключом
        и новая реплика до ответа на предыдущую дают HTTP 409. Снимает только
        автоматическую паузу инициативных сообщений, сохраняя пользовательскую.
        """
        async with self._tx(sessions) as session:
            settings = await ensure_user(session, user_id)
            await owned_conversation(session, user_id, conversation_id)
            now = utcnow()
            previous = await session.scalar(
                select(Message).where(
                    Message.user_id == user_id, Message.client_message_id == body.client_message_id
                )
            )
            if previous:
                if previous.conversation_id != conversation_id or previous.content != body.content:
                    raise HTTPException(
                        409, "client_message_id уже использован для другого сообщения"
                    )
                assistant = await self._reply_to(session, user_id, conversation_id, previous.id)
                if assistant:
                    return "replay", self._replay(previous, assistant)
                if self._fresh(previous, now):
                    return "wait", previous.id
                # Обработка была прервана: продолжаем её для уже сохранённой реплики.
                message = previous
            else:
                busy = await session.scalar(
                    select(Message)
                    .where(
                        Message.user_id == user_id,
                        Message.conversation_id == conversation_id,
                        Message.role == "user",
                        Message.created_at >= now - self.processing_timeout,
                    )
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
                if busy is not None and self._fresh(busy, now):
                    # Порядок реплик в беседе сохраняется: новый текст принимается после ответа.
                    raise HTTPException(
                        409,
                        {
                            "code": "previous_message_processing",
                            "message": "Предыдущее сообщение ещё обрабатывается. Повторите через несколько секунд.",
                        },
                    )
                message = Message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="user",
                    origin="user_input",
                    content=body.content,
                    client_message_id=body.client_message_id,
                    read_at=now,
                )
                session.add(message)
            message.details = {
                **(message.details or {}),
                "processing": "pending",
                "processing_started_at": now.isoformat(),
            }
            if settings.proactive_pause_reason != "user":
                # Автоматическая пауза снимается ответом; пауза, заданная пользователем, сохраняется.
                settings.proactive_paused_until = settings.proactive_pause_reason = None
            await session.flush()
            previous_turns = []
            if len(body.content.strip()) <= 100:
                prior = list(
                    (
                        await session.scalars(
                            select(Message)
                            .where(
                                Message.user_id == user_id,
                                Message.conversation_id == conversation_id,
                                Message.id != message.id,
                                Message.created_at >= now - timedelta(hours=72),
                            )
                            .order_by(Message.created_at.desc(), Message.id.desc())
                            .limit(2)
                        )
                    ).all()
                )
                previous_turns = [
                    {"role": m.role, "content": m.content[-1200:]} for m in reversed(prior)
                ]
            snapshot = SimpleNamespace(
                previous_turns=previous_turns,
                message_id=message.id,
                consent=settings.external_ai_consent,
                country=settings.country_code,
                preferences=dict(settings.ai_preferences or {}),
            )
        return "new", snapshot

    async def _await_result(self, sessions, user_id, conversation_id, body, message_id):
        """Ждёт ответ на уже принятую реплику, проверяя БД каждые 0,25 секунды.

        Готовый ответ возвращает повторно; прерванную или просроченную обработку
        продолжает через send. По общему тайм-ауту возвращает HTTP 409, при удалении
        реплики — 404. Между проверками соединение не удерживается.
        """
        deadline = perf_counter() + self.processing_timeout.total_seconds()
        while perf_counter() < deadline:
            await asyncio.sleep(POLL_SECONDS)
            async with sessions() as session:
                assistant = await self._reply_to(session, user_id, conversation_id, message_id)
                if assistant:
                    return self._replay(await session.get(Message, message_id), assistant)
                message = await session.get(Message, message_id)
                if message is None:
                    raise HTTPException(404, "Беседа не найдена")
                if not self._fresh(message, utcnow()):
                    # Первая обработка прервалась: этот запрос продолжит её.
                    return await self.send(sessions, user_id, conversation_id, body)
        raise HTTPException(
            409,
            {"code": "message_processing", "message": "Сообщение ещё обрабатывается."},
        )

    # ------------------------------------------------------------------ public
    async def send(self, sessions, user_id, conversation_id, body):
        """Принимает реплику и возвращает сохранённую пару сообщений с отчётом обработки.

        Повтор с тем же client_message_id читает готовый результат или ждёт первую
        обработку. При ошибке либо отмене запроса помечает незавершённую реплику
        interrupted, чтобы следующий повтор мог продолжить её сразу.
        """
        status, accepted = await self._accept(sessions, user_id, conversation_id, body)
        if status == "replay":
            return accepted
        if status == "wait":
            return await self._await_result(sessions, user_id, conversation_id, body, accepted)
        try:
            return await self._process(sessions, user_id, conversation_id, body, accepted)
        except BaseException:
            # Сбой или обрыв соединения клиента: снимаем отметку обработки, чтобы повтор
            # с тем же client_message_id продолжил работу сразу, а не ждал тайм-аута.
            await asyncio.shield(self._interrupt(sessions, accepted.message_id))
            raise

    async def _interrupt(self, sessions, message_id):
        """Помечает ещё не завершённую реплику interrupted в отдельной транзакции.

        Ошибка очистки записывается фиксированным кодом в журнал и не перекрывает
        исходное исключение обработки.
        """
        try:
            async with self._tx(sessions) as session:
                message = await session.get(Message, message_id)
                if message is not None and message.details.get("processing") == "pending":
                    message.details = {**message.details, "processing": "interrupted"}
        except Exception:
            log.error("chat_interrupt_cleanup_failed")

    async def _summarize(self, boundary, safety, previous, messages):
        """Получает резюме ранних реплик через LLM по обезличенному тексту.

        Восстанавливает псевдонимы, проверяет программные запреты и разрешённые
        контакты, ограничивает результат 6000 символами. При недоступности модели
        или отклонённом резюме возвращает None; записей в БД не делает.
        """
        request = summary_request(previous, messages)
        try:
            raw = await self.llm.generate(
                [request[0], {"role": "user", "content": boundary.mask(request[1]["content"])}]
            )
        except ProviderUnavailable:
            return None
        text = boundary.unmask(str(raw)).strip()
        # Резюме — тоже вывод модели: диагнозы, назначения и выдуманные контакты в контекст не идут.
        if not text or not safety.deterministic_output(text) or safety.unverified_contacts(text):
            return None
        return text[:6000]

    async def _process(self, sessions, user_id, conversation_id, body, accepted):
        """Выполняет анализ, выбор ответа, генерацию и сохранение результата реплики.

        Явные правила проверяют исходный текст, внешние модели — обезличенный.
        Срочный маршрут обходит модели; недоступность анализа и ошибки генерации
        ведут к заданным шаблонам. Генерация, свёртка контекста и проверка ответа
        выполняются вне транзакций. В конце атомарно сохраняет ответ, события,
        память и отчёт; параллельно сохранённый ответ возвращает как replay.
        """
        message_id, country = accepted.message_id, accepted.country

        report = DecisionReport(self.models)
        report.stage("authorization", "passed", ["jwt_verified", "conversation_ownership_verified"])
        report.stage("persistence", "passed", ["client_message_id_new", "user_message_saved"])
        jev = self.jev
        consent_missing = self.settings.ai_mode == "live" and not accepted.consent
        if consent_missing:
            jev = DisabledJevProvider()
        safety = SafetyGuard(self.settings, jev)
        # Единая граница приватности запроса: одна и та же замена во всех исходящих частях.
        boundary = boundary_for(self.settings, accepted.preferences, safety.allowed_contacts()[0])

        # ---- вне транзакции: правила (по исходному тексту) и классификация (по обезличенному)
        started = perf_counter()
        lexicon = safety.deterministic(body.content)
        deterministic = lexicon.route
        report.stage(
            "input_rules",
            "passed",
            route=deterministic,
            risks=lexicon.risks,
            policy_version=safety.rules["version"],
            lexicon_version=LEXICON_VERSION,
        )
        input_called = (
            deterministic != "urgent"
            and not consent_missing
            and self.settings.ai_mode != "disabled"
        )
        if deterministic == "urgent":
            analysis = EmotionAnalysis(available=False, scope="unavailable", input_calls=0)
        else:
            analysis = await EmotionAnalyzer(jev).analyze(
                boundary.mask(body.content), boundary.mask_messages(accepted.previous_turns)
            )
        reasons = (
            ["urgent_bypass"]
            if deterministic == "urgent"
            else ["external_ai_consent_missing"]
            if consent_missing
            else ["ai_disabled"]
            if self.settings.ai_mode == "disabled"
            else ["irrelevant_message_full_analysis_skipped"]
            if analysis.scope == "irrelevant"
            else ["classifier_completed"]
            if analysis.available
            else ["classifier_unavailable"]
        )
        report.stage(
            "jev_analysis",
            "completed" if analysis.available else "unavailable" if input_called else "skipped",
            reasons,
            called=input_called,
            scope=analysis.scope,
            gate_score=analysis.gate_score,
            input_calls=analysis.input_calls if input_called else 0,
            error_code=analysis.error_code,
            provider=self.models["jev"]["provider"],
            model=analysis.model_version if analysis.available else self.models["jev"]["model"],
            intent=analysis.intent if analysis.available else None,
            markers=analysis.markers,
            action_threshold=self.settings.jev_threshold,
            review_threshold=self.settings.jev_review_threshold,
            sensitive_review_threshold=self.settings.jev_sensitive_review_threshold,
            triggered_markers=[
                k for k, v in analysis.markers.items() if v >= self.settings.jev_threshold
            ],
            source=analysis.source if analysis.available else None,
            not_a_diagnosis=True,
        )
        report.data["timings_ms"]["input_analysis"] = report.elapsed(started)
        decision = safety.assess_input(body.content, analysis)
        report.stage(
            "safety_route",
            "completed",
            decision.reasons,
            route=decision.route,
            risks=decision.risks,
        )

        # ---- транзакция 2: наблюдения, контекст, решение
        plan = await self._decide(
            sessions,
            user_id,
            conversation_id,
            message_id,
            analysis,
            decision,
            safety,
            report,
            accepted.preferences,
            boundary.enabled,
        )
        route, state = plan.route, plan.state
        uses_llm = route != "urgent" and analysis.available

        # ---- вне транзакции: контроль контекста, генерация и проверка ответа
        started = perf_counter()
        manager = ChatContextManager(plan.budget)
        fitted = manager.fit(plan.system_prompt, plan.context, plan.history, plan.summary)
        new_summary = folded = None
        summary_calls = 0
        if uses_llm and fitted.to_summarize:
            summary_calls = 1
            folded = fitted.to_summarize
            before_summary = fitted.report.original_prompt_tokens
            new_summary = await self._summarize(boundary, safety, plan.summary, folded)
            if new_summary:
                fitted = manager.fit(
                    plan.system_prompt,
                    plan.context,
                    plan.history[folded[-1]["index"] + 1 :],
                    new_summary,
                )
            else:
                # Резюме получить не удалось: остаётся усечение с явной пометкой.
                folded = None
                fitted = manager.fit(
                    plan.system_prompt,
                    plan.context,
                    plan.history,
                    plan.summary,
                    summary_available=False,
                )
        context_report = fitted.report
        if summary_calls:
            # «Было» — размер до свёртки истории, а не уже сжатого повторного расчёта.
            context_report.original_prompt_tokens = max(
                before_summary, context_report.original_prompt_tokens
            )
        if new_summary:
            context_report.summarized_messages = len(folded)
            context_report.applied_steps.append(
                {"step": "summary_generated", "messages": len(folded), "llm_calls": 1}
            )
        elif summary_calls:
            context_report.applied_steps.append({"step": "summary_failed", "llm_calls": 1})
        report.data["timings_ms"]["context_control"] = report.elapsed(started)

        started = perf_counter()
        origin, fallback, llm_calls, output_checks, events = "reactive", None, 0, [], []
        addendum = offered = False
        usage = None
        if route == "urgent":
            kind = safety.urgent_kind(decision)
            text, origin, fallback = (
                safety.template(kind, country),
                "safety_template",
                f"{kind}_template",
            )
            # Код правила, не текст сообщения.
            events.append(
                (",".join(decision.risks)[:120] or "possible_immediate_danger", f"{kind}_template")
            )
        elif not analysis.available:
            text, origin, fallback = (
                safety.template("limited", country),
                "safety_template",
                "limited_mode",
            )
            events.append(("classifier_unavailable", "limited_mode"))
        elif context_report.verdict == "over_budget":
            # Сжатие выключено или исчерпано: запрос сверх окна модели не отправляется.
            text, origin, fallback = (
                safety.template("context_overflow", country),
                "safety_template",
                "context_over_budget",
            )
            events.append(("context_over_budget", "context_overflow_template"))
        else:
            prompt = [fitted.messages[0], *boundary.mask_messages(fitted.messages[1:])]
            try:
                llm_calls += 1
                text = await self.llm.generate(prompt)
                usage = getattr(text, "usage", None)
                check = await safety.evaluate_output(text, route, state.referral_required)
                output_checks.append(check.model_dump())
                if not check.safe:
                    events.append(("unsafe_output", "repair_once"))
                    llm_calls += 1
                    text = await self.llm.generate(prompt, repair=True)
                    check = await safety.evaluate_output(text, route, state.referral_required)
                    output_checks.append(check.model_dump())
                    if not check.safe:
                        text, origin, fallback = (
                            safety.template(
                                "professional" if route != "normal" else "output_blocked", country
                            ),
                            "safety_template",
                            "output_safety_fallback",
                        )
                        events.append(("unsafe_output", "fallback"))
            except ProviderUnavailable:
                # При основании для обращения технический сбой не должен скрыть рекомендацию.
                kind = "technical" if route == "normal" else "professional"
                text, origin, fallback = (
                    safety.template(kind, country),
                    "safety_template",
                    "llm_unavailable",
                )
                events.append(("llm_unavailable", f"{kind}_template"))
            if origin == "reactive":
                # Проверки шли по обезличенному тексту; пользователю возвращаются её слова.
                text = boundary.unmask(str(text))
            if decision.self_harm and origin == "reactive":
                # Рекомендация при мыслях о самоповреждении добавляется кодом, а не моделью.
                text = text.rstrip() + "\n\n" + safety.template("self_harm_addendum", country)
                addendum = True
            if route != "normal":
                events.append(
                    (
                        ",".join(decision.risks)[:120] or "professional_support_needed",
                        "recommend_professional",
                    )
                )
            offered = bool(state.screening_offer and origin == "reactive" and route == "normal")
        report.stage(
            "privacy",
            "applied" if boundary.enabled else "disabled",
            ["outbound_text_pseudonymized"] if boundary.enabled else ["anonymization_disabled"],
            **boundary.report.as_dict(),
        )
        report.stage(
            "context",
            context_report.verdict,
            ["bounded_history_and_personal_context", "user_content_is_untrusted_data"],
            active_complaints=plan.active_complaints,
            sent_to_llm=bool(llm_calls),
            provider_prompt_tokens=(usage or {}).get("prompt_tokens"),
            provider_completion_tokens=(usage or {}).get("completion_tokens"),
            summary_llm_calls=summary_calls,
            **context_report.as_event(),
        )
        report.data["timings_ms"]["generation_and_output_guard"] = report.elapsed(started)
        report.stage(
            "llm_generation",
            "skipped"
            if not llm_calls
            else "failed"
            if fallback == "llm_unavailable"
            else "completed",
            [fallback] if not llm_calls or fallback == "llm_unavailable" else ["provider_called"],
            provider=self.models["llm"]["provider"],
            model=self.models["llm"]["model"],
            calls=llm_calls,
            repair_attempted=llm_calls > 1,
        )
        report.stage(
            "output_safety",
            "skipped"
            if not output_checks
            else "passed"
            if output_checks[-1]["safe"]
            else "blocked",
            ["configured_template_used"] if not output_checks else output_checks[-1]["reasons"],
            attempts=output_checks,
            output_threshold=self.settings.jev_output_threshold,
            self_harm_addendum=addendum,
            referral_required=state.referral_required,
        )

        # ---- транзакция 3: сохранение ответа
        async with self._tx(sessions) as session:
            settings_row = await ensure_user(session, user_id)
            conversation = await owned_conversation(session, user_id, conversation_id)
            message = await session.get(Message, message_id)
            existing = await self._reply_to(session, user_id, conversation_id, message_id)
            if existing:
                # Параллельное продолжение уже сохранило ответ; второй не создаётся.
                return self._replay(message, existing)
            for risk_type, action in events:
                safety.record(session, user_id, message_id, risk_type, route, action)
            if offered:
                # Generation runs without a DB lock: completion, snooze, stage and offers
                # may have changed. Recheck under the same user lock before delivery.
                offer_time = utcnow()
                schedules = list(
                    (
                        await session.scalars(
                            select(AssessmentSchedule).where(AssessmentSchedule.user_id == user_id)
                        )
                    ).all()
                )
                schedule = next(
                    (r for r in schedules if r.questionnaire_code == state.screening_offer), None
                )
                new_risk = await session.scalar(
                    select(SafetyEvent.id)
                    .where(
                        SafetyEvent.user_id == user_id,
                        SafetyEvent.created_at >= plan.decision_at,
                        SafetyEvent.risk_route.in_(("urgent", "recommend_professional")),
                    )
                    .limit(1)
                )
                offered = (
                    settings_row.pregnancy_id == plan.pregnancy_id
                    and settings_row.birth_date == plan.birth_date
                    and perinatal_context(settings_row, offer_time).stage == state.perinatal_stage
                    and new_risk is None
                    and all(
                        r.last_offered_at is None
                        or r.last_offered_at <= offer_time - timedelta(hours=24)
                        for r in schedules
                    )
                    and not (
                        schedule and schedule.snoozed_until and schedule.snoozed_until > offer_time
                    )
                    and not (
                        schedule
                        and schedule.last_completed_at
                        and schedule.last_completed_at >= plan.decision_at
                    )
                )
            if offered:
                verb = "продолжить начатый" if state.screening_attempt_id else "пройти"
                text = text.rstrip() + (
                    f"\n\nЕсли захочешь, можно {verb} необязательный опросник "
                    f"{SCALE_NAMES[state.screening_offer]} в разделе «Опросники». "
                    "Это самооценка самочувствия, а не диагноз."
                )
                await ScreeningScheduler(self.settings).record(
                    session, user_id, state.screening_offer, "offered"
                )
            else:
                state.screening_offer = state.screening_reason = state.screening_attempt_id = None
            report.stage(
                "screening_offer",
                "offered" if offered else "skipped",
                ["invitation_delivered"] if offered else ["no_eligible_invitation_at_delivery"],
                code=state.screening_offer,
            )
            if boundary.enabled:
                # Имена, названные в этой реплике, запоминаются для следующих запросов.
                known = list(settings_row.ai_preferences.get("learned_names") or [])
                fresh = [n for n in boundary.learned_names() if n not in known]
                if fresh:
                    settings_row.ai_preferences = {
                        **settings_row.ai_preferences,
                        "learned_names": [*known, *fresh][-LEARNED_NAMES_KEPT:],
                    }
            if new_summary:
                conversation.context_summary = new_summary
                conversation.context_summary_until = plan.history_times[folded[-1]["index"]]
                conversation.context_summary_messages = (
                    conversation.context_summary_messages or 0
                ) + len(folded)
            answer = Message(
                user_id=user_id,
                conversation_id=conversation_id,
                role="assistant",
                origin=origin,
                content=text,
                protocol_id=plan.protocol_id,
                protocol_version=plan.protocol_version,
                details={
                    "reply_to": str(message_id),
                    "state": state.model_dump(),
                    "analysis_available": analysis.available,
                    "safety_policy_version": safety.rules["version"],
                    "clinical_protocol_approved": plan.protocol_approved,
                    "referral_made": safety.has_referral(text),
                },
            )
            session.add(answer)
            conversation.updated_at = utcnow()
            unread = await session.scalars(
                select(Message).where(
                    Message.user_id == user_id,
                    Message.origin == "proactive",
                    Message.read_at.is_(None),
                )
            )
            for proactive in unread:
                proactive.read_at = utcnow()
            # Маршрут сохраняется на реплике: срочные реплики не пересказываются в памяти.
            message.details = {
                **{
                    k: v
                    for k, v in message.details.items()
                    if k not in ("processing", "processing_started_at")
                },
                "route": route,
            }
            memory = MemoryManager()
            prior_cursor = (await memory.get(session, user_id)).last_summarized_message_id
            await memory.update(session, user_id, message, analysis, self.settings.jev_threshold)
            updated = (
                await memory.get(session, user_id)
            ).last_summarized_message_id != prior_cursor
            report.stage(
                "memory",
                "updated" if updated else "unchanged",
                ["summary_refreshed_from_user_sources"] if updated else ["summary_refresh_not_due"],
            )
            answer.details = {
                **answer.details,
                "decision_report": report.finish(
                    route=route,
                    origin=origin,
                    protocol=plan.protocol_id,
                    fallback=fallback,
                    model_revision=self.models.get("revision", 0),
                    llm_calls=llm_calls,
                    summary_llm_calls=summary_calls,
                    jev_input_called=input_called,
                    jev_output_calls=sum(c["jev_called"] for c in output_checks),
                ),
            }
            await session.flush()
            result = {
                "user_message": public(message),
                "assistant_message": public(answer),
                "replayed": False,
            }
        # Only non-sensitive operational metadata goes to application stdout.
        log.info(
            "chat_processed trace_id=%s total_ms=%s stages=%s",
            report.data["trace_id"],
            report.data["total_ms"],
            len(report.data["stages"]),
        )
        return result

    # ------------------------------------------------------------------ phase 2
    async def _decide(
        self,
        sessions,
        user_id,
        conversation_id,
        message_id,
        analysis,
        decision,
        safety,
        report,
        preferences,
        privacy_enabled,
    ):
        """Собирает план ответа в транзакции с блокировкой профиля.

        Сохраняет наблюдения, читает свежие шкалы, память и историю, вычисляет
        потребность в поддержке и учитывает интервал напоминания о специалисте.
        После закрытия транзакции собирает системную инструкцию, протокол и бюджет
        контекста. Возвращает план; к внешним провайдерам не обращается.
        """
        route = decision.route
        async with self._tx(sessions) as session:
            settings = await ensure_user(session, user_id)
            conversation = await owned_conversation(session, user_id, conversation_id)
            now = utcnow()
            message = await session.get(Message, message_id)
            perinatal = perinatal_context(settings, now)
            complaint_manager = ComplaintManager()
            await complaint_manager.observe(
                session,
                user_id,
                message,
                analysis,
                self.settings.jev_threshold,
                settings.pregnancy_id,
            )
            report.stage(
                "complaints",
                "completed" if analysis.available else "skipped",
                ["observations_stored_without_inferred_resolution"]
                if analysis.available
                else ["no_fabricated_observations"],
                observed_categories=[
                    k
                    for k, v in analysis.markers.items()
                    if k in CATEGORIES and v >= self.settings.jev_threshold
                ],
            )
            history = await assessment_history(session, user_id)
            screenings = await recent_results(
                session,
                user_id,
                now,
                safety.rules.get("screening_result_valid_days", 30),
                history=history,
            )
            memory = MemoryManager()
            context, complaints = await memory.context(session, user_id, perinatal)
            patient_context = await PatientStateBuilder(self.settings).build(
                session,
                user_id,
                analysis,
                perinatal,
                message_id,
                now,
                history=history,
                screening=screenings,
            )
            recurrence = patient_context["long"]["patterns"]
            context["patient_state"] = prompt_patient_state(patient_context)
            rows = await memory.history(
                session, user_id, conversation_id, conversation.context_summary_until
            )
            # Do not pay twice for source excerpts already present verbatim in the dialogue.
            stored_memory = await memory.get(session, user_id)
            history_ids = {row["id"] for row in rows}
            if stored_memory.facts:
                remaining = [
                    f
                    for f in stored_memory.facts
                    if f.get("source") != "user_edit" and f.get("message_id") not in history_ids
                ]
                context["user_memory"] = "\n".join(
                    f["text"] for f in stored_memory.facts if f.get("source") == "user_edit"
                )
                context["summary"] = "\n".join(
                    f"{f['observed_at'][:10]}: {f['text']}" for f in remaining
                )
            summary = conversation.context_summary or ""
            evaluator = StateEvaluator(safety.rules)
            state = evaluator.evaluate(
                analysis,
                complaints,
                route,
                threshold=self.settings.jev_threshold,
                perinatal=perinatal,
                recurrence=recurrence,
                screening=screenings,
                patient_context=patient_context,
            )
            if route != "urgent":
                actions = await ScreeningScheduler(self.settings).next_actions(
                    session, user_id, now, plan=patient_context["assessments"], screening=screenings
                )
                last_offers = [
                    r.get("last_offered_at")
                    for r in patient_context["assessments"]
                    if r.get("last_offered_at")
                ]
                chat_cooldown = timedelta(hours=24)
                if (
                    not last_offers
                    or max(datetime.fromisoformat(t) for t in last_offers) <= now - chat_cooldown
                ):
                    offer = next((a for a in actions if a["type"] == "questionnaire"), None)
                    if offer:
                        state.screening_offer = offer["code"]
                        state.screening_reason = offer["reason"]
                        state.screening_attempt_id = offer.get("attempt_id")
            standing_reason = None
            if state.recommend_specialist and route == "normal":
                # Постоянное основание (длительные жалобы, результат шкалы) напоминается
                # с интервалом, а не превращает каждый ответ в направление к врачу.
                reminder = timedelta(days=safety.rules.get("referral_reminder_days", 7))
                last_referral = await self.last_referral(session, user_id, now - reminder)
                new_safety = any(
                    r.get("safety_flags")
                    and (
                        last_referral is None
                        or datetime.fromisoformat(r["safety_at"]) > last_referral
                    )
                    for r in screenings
                )
                if last_referral is None or new_safety:
                    route = state.safety_route = "recommend_professional"
                    standing_reason = "standing_referral_due"
                else:
                    standing_reason = "standing_referral_recently_made"
            state.referral_required = route == "recommend_professional"
            if state.recommend_specialist or route != "normal":
                state.screening_offer = state.screening_reason = state.screening_attempt_id = None
            content, country = message.content, settings.country_code
        state_reasons = (
            [state.support_basis]
            if state.primary_support_need != "general"
            else ["no_support_marker_above_threshold"]
        )
        primary_value = analysis.markers.get(state.primary_support_need, 0)
        if (
            state.primary_support_need != "general"
            and state.support_basis == "highest_marker_above_action_threshold"
            and sum(analysis.markers.get(key, -1) == primary_value for key in CATEGORIES) > 1
        ):
            state_reasons.append("equal_markers_fixed_category_priority")
        if state.distress_pattern in ("recurring", "persistent"):
            state_reasons.append("recurring_complaint_observations")
        state_reasons += [f"specialist:{reason}" for reason in state.specialist_reasons]
        if standing_reason:
            state_reasons.append(standing_reason)
        if analysis.markers.get("functional_impact", 0) >= self.settings.jev_threshold:
            state_reasons.append("functional_impact_above_threshold")
        if state.postpartum:
            state_reasons.append("postpartum_context_present")
        report.stage(
            "state_evaluation",
            "completed",
            state_reasons,
            category_priority=list(CATEGORIES),
            **state.model_dump(),
        )
        selector = SupportProtocolSelector(self.settings)
        protocol, postpartum = selector.select(state)
        protocol_id = protocol["id"] if protocol else "safe_baseline"
        protocol_reason = (
            "urgent_protocol_bypass"
            if route == "urgent"
            else "approved_protocol_matched"
            if protocol
            else "requested_protocol_not_approved_or_missing"
        )
        # Линия помощи попадает в инструкцию только по поводу: есть сигнал риска, нужна
        # рекомендация специалиста либо она сама просит контакт.
        contact_requested = bool(CONTACT_REQUEST.search(content.lower()))
        allow_contact = state.referral_required or bool(decision.risks) or contact_requested
        report.stage(
            "protocol_selection",
            "skipped" if route == "urgent" else "completed",
            [protocol_reason],
            requested=state.support_protocol,
            applied=protocol_id if route != "urgent" else None,
            version=protocol["version"] if protocol else "1.0",
            approved=protocol is not None,
            postpartum_applied=postpartum["id"] if postpartum else None,
            support_contact_allowed=allow_contact,
        )
        return SimpleNamespace(
            decision_at=now,
            pregnancy_id=settings.pregnancy_id,
            birth_date=settings.birth_date,
            route=route,
            state=state,
            system_prompt=selector.system_prompt(
                state,
                protocol,
                postpartum,
                safety.support_text(country) if allow_contact else "",
                privacy_note=privacy_enabled,
            ),
            context=context,
            history=[{**row, "index": index} for index, row in enumerate(rows)],
            history_times=[datetime.fromisoformat(row["at"]) for row in rows],
            summary=summary,
            budget=ContextBudget.from_settings(
                self.settings, preferences, self.settings.llm_max_output_tokens
            ),
            active_complaints=len(complaints),
            protocol_id=protocol_id,
            protocol_version=protocol["version"] if protocol else "1.0",
            protocol_approved=protocol is not None,
        )


CONTACT_REQUEST = re.compile(
    r"телефон|позвонить|горяч\w+ лини|лини\w+ (?:помощи|доверия)|куда (?:мне |можно )?обратиться|"
    r"с кем (?:мне |можно )?поговорить|не с кем поговорить|контакт"
)


class _UserTransaction:
    """Короткая транзакция. Блокировку пользователя берёт вызывающий через ensure_user."""

    def __init__(self, sessions):
        """Сохраняет фабрику асинхронных сессий для короткой транзакции."""
        self.sessions = sessions

    async def __aenter__(self):
        """Открывает сессию и транзакцию и передаёт сессию вызывающему коду."""
        self.session = self.sessions()
        await self.session.__aenter__()
        self.transaction = self.session.begin()
        await self.transaction.__aenter__()
        return self.session

    async def __aexit__(self, *exc):
        """Фиксирует или откатывает транзакцию и закрывает сессию даже при ошибке."""
        try:
            await self.transaction.__aexit__(*exc)
        finally:
            await self.session.__aexit__(*exc)

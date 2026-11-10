"""Маршрутизация и основания рекомендации специалиста. Инженерные проверки, не клиническая валидация."""

import asyncio
import os
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, update

from emo_core.core.auth import DEMO_USERS
from emo_core.core.exceptions import ProviderUnavailable
from emo_core.db.base import utcnow
from emo_core.db.models import (
    AssessmentSchedule,
    MessageSignal,
    SafetyEvent,
    UserSettings,
)
from emo_core.db.models import TestAttempt as QuestionnaireAttempt
from emo_core.schemas.emotion import EmotionAnalysis
from emo_core.services.perinatal import perinatal_context
from emo_core.services.screening_scheduler import ScreeningScheduler
from emo_core.services.state_evaluator import StateEvaluator
from tests.conftest import conversation, send
from tests.test_questionnaire_workflow import MATERIALS, catalog, minimum_answers

P = "/api/v1/emo"
ANNA = DEMO_USERS["anna"]
NOW = datetime(2026, 11, 10, 12, tzinfo=UTC)


def stages(response):
    """Извлекает этапы и итоговый отчёт решения из HTTP-ответа чата."""
    report = response.json()["assistant_message"]["metadata"]["decision_report"]
    return {s["stage"]: s for s in report["stages"]}, report


async def complete(client, auth, code, answers):
    """Завершает шкалу через API и возвращает проверенный успешный результат."""
    q = await catalog(client, auth, code)
    response = await client.post(
        P + "/questionnaire-attempts",
        headers=auth["anna"],
        json={"questionnaire_id": q["id"], "answers": answers, "complete": True},
    )
    assert response.status_code == 200, response.text
    return response.json()


def epds_answers(score):
    """Собирает набор индексов ответов EPDS с заданной суммой для сценария маршрутизации."""
    answers, remaining = minimum_answers("EPDS"), score
    for item in MATERIALS["EPDS"]["definition"]["items"][:9]:
        value = min(3, remaining)
        answers[item["id"]] = item["score_map"].index(value)
        remaining -= value
    return answers


# ---------------------------------------------------------------- perinatal context
def user(**values):
    """Создаёт объект перинатальных настроек с тестовыми значениями и переопределениями."""
    base = dict(
        pregnancy_outcome=None,
        birth_date=None,
        estimated_due_date=None,
        gestational_week=None,
        gestational_week_recorded_at=None,
    )
    return SimpleNamespace(**{**base, **values})


def test_gestational_week_follows_calendar():
    """Проверяет расчёт срока по ПДР, дате записи недели и устаревшему сроку."""
    by_due_date = perinatal_context(user(estimated_due_date=date(2027, 1, 26)), NOW)
    assert (by_due_date.stage, by_due_date.gestational_week, by_due_date.trimester) == (
        "pregnant",
        29,
        3,
    )
    recorded = user(gestational_week=20, gestational_week_recorded_at=NOW - timedelta(weeks=9))
    assert perinatal_context(recorded, NOW).gestational_week == 29
    stale = user(gestational_week=30, gestational_week_recorded_at=NOW - timedelta(weeks=20))
    assert perinatal_context(stale, NOW).stage == "unknown"


def test_postpartum_is_bounded_and_loss_is_respected():
    """Проверяет границу послеродового периода и отдельный контекст после потери."""
    assert perinatal_context(user(birth_date=NOW - timedelta(days=40)), NOW).postpartum_weeks == 5
    assert perinatal_context(user(birth_date=NOW - timedelta(days=500)), NOW).stage == "unknown"
    loss = user(pregnancy_outcome="loss", gestational_week=12, gestational_week_recorded_at=NOW)
    assert perinatal_context(loss, NOW).stage == "after_loss"


async def test_settings_validate_and_anchor_pregnancy_dates(client, auth, sessions):
    """Проверяет валидацию перинатальных дат и фиксацию момента записи срока."""
    future = (utcnow() + timedelta(days=30)).isoformat()
    assert (
        await client.patch(P + "/settings", headers=auth["anna"], json={"birth_date": future})
    ).status_code == 422
    response = await client.patch(
        P + "/settings", headers=auth["anna"], json={"gestational_week": 20}
    )
    assert response.status_code == 200 and response.json()["gestational_week_recorded_at"]
    assert (
        await client.patch(
            P + "/settings", headers=auth["anna"], json={"pregnancy_outcome": "invented"}
        )
    ).status_code == 422


async def test_loss_context_reaches_prompt_without_baby_talk(client, auth, app):
    """Проверяет передачу сведений о потере беременности в правила ответа модели."""

    class Spy:
        system = ""

        async def generate(self, messages, repair=False):
            """Запоминает системные правила и возвращает нейтральный ответ для проверки контекста."""
            Spy.system = messages[0]["content"]
            return "Мне очень жаль. Я рядом, если захочешь рассказать."

    app.state.llm = Spy()
    await client.patch(P + "/settings", headers=auth["anna"], json={"pregnancy_outcome": "loss"})
    cid = await conversation(client, auth)
    steps, _ = stages(await send(client, auth, cid, "Мне очень плохо"))
    assert steps["state_evaluation"]["facts"]["perinatal_stage"] == "after_loss"
    assert "завершилась потерей" in Spy.system


# ---------------------------------------------------------------- safety in the pipeline
async def test_self_harm_ideation_gets_code_appended_guidance(client, auth, sessions):
    """Проверяет обязательное добавление контактов помощи к ответу при соответствующих мыслях."""
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Иногда думаю, что не хочу жить")
    answer = response.json()["assistant_message"]
    steps, report = stages(response)
    assert answer["origin"] == "reactive" and report["result"]["route"] == "recommend_professional"
    assert "112" in answer["content"] and "специалист" in answer["content"]
    assert steps["output_safety"]["facts"]["self_harm_addendum"] is True
    assert answer["metadata"]["referral_made"] is True
    async with sessions() as s:
        event = await s.scalar(
            select(SafetyEvent).where(SafetyEvent.risk_route == "recommend_professional")
        )
        assert "self_harm_ideation" in event.risk_type and event.details == {}


async def test_obstetric_warning_sign_uses_medical_template_without_models(client, auth, app):
    """Проверяет срочный медицинский шаблон и обход моделей при явном тревожном признаке."""

    class Never:
        async def generate(self, *a, **kw):
            """Проваливает тест при недопустимом вызове генератора на срочном маршруте."""
            pytest.fail("LLM must not run for an urgent medical sign")

        async def analyze(self, *a, **kw):
            """Проваливает тест при недопустимом вызове Jev на срочном маршруте."""
            pytest.fail("Classifier must not delay an urgent medical sign")

    app.state.llm = app.state.jev = Never()
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Малыш не шевелится с самого утра")
    answer = response.json()["assistant_message"]
    assert answer["origin"] == "safety_template"
    assert "скорую" in answer["content"] and "103" in answer["content"]
    assert stages(response)[1]["result"]["fallback"] == "medical_urgent_template"


async def test_llm_failure_keeps_referral_when_route_requires_it(client, auth, app):
    """Проверяет сохранение рекомендации специалиста при сбое генерации."""

    class Broken:
        async def generate(self, messages, repair=False):
            """Имитирует недоступность генератора на маршруте рекомендации."""
            raise ProviderUnavailable("llm_unavailable")

    app.state.llm = Broken()
    cid = await conversation(client, auth)
    answer = (
        await send(client, auth, cid, "Какие таблетки от тревоги можно мне принимать?")
    ).json()["assistant_message"]
    assert "врачом" in answer["content"] and "временно недоступен" not in answer["content"]


# ---------------------------------------------------------------- frequency vs persistence
async def test_many_mentions_in_one_day_do_not_create_referral(client, auth):
    """Проверяет, что частые упоминания за один день не доказывают длительность трудностей."""
    cid = await conversation(client, auth)
    for _ in range(5):
        response = await send(client, auth, cid, "Мне тревожно")
    steps, report = stages(response)
    assert report["result"]["route"] == "normal"
    assert steps["state_evaluation"]["facts"]["recommend_specialist"] is False
    assert steps["state_evaluation"]["facts"]["distress_pattern"] == "unknown"


async def test_persistent_complaint_referral_is_reminded_not_repeated(client, auth, sessions):
    """Проверяет интервал напоминания о специалисте при сохраняющейся жалобе."""
    cid = await conversation(client, auth)
    for _ in range(3):
        await send(client, auth, cid, "Мне тревожно")
    async with sessions() as s, s.begin():
        rows = list(
            (
                await s.scalars(
                    select(MessageSignal)
                    .where(MessageSignal.signal_code == "anxiety")
                    .order_by(MessageSignal.observed_at)
                )
            ).all()
        )
        for row, days in zip(rows, (16, 8, 1), strict=True):
            row.observed_at = utcnow() - timedelta(days=days)
    first = await send(client, auth, cid, "Мне снова тревожно")
    steps, report = stages(first)
    facts = steps["state_evaluation"]["facts"]
    assert facts["distress_pattern"] == "persistent"
    assert facts["specialist_reasons"] == ["persistent_complaint"] and facts["referral_required"]
    assert "standing_referral_due" in steps["state_evaluation"]["reasons"]
    assert report["result"]["route"] == "recommend_professional"
    assert "специалист" in first.json()["assistant_message"]["content"]
    second = await send(client, auth, cid, "И сегодня тревожно")
    steps, report = stages(second)
    assert report["result"]["route"] == "normal"
    assert steps["state_evaluation"]["facts"]["recommend_specialist"] is True
    assert steps["state_evaluation"]["facts"]["referral_required"] is False
    assert "standing_referral_recently_made" in steps["state_evaluation"]["reasons"]


def test_estrangement_matters_only_when_it_persists():
    """Проверяет различие разового и устойчивого сигнала отчуждения."""
    analysis = EmotionAnalysis(markers={"maternal_estrangement": 0.9})
    once = StateEvaluator().evaluate(
        analysis, [], "normal", recurrence={"maternal_estrangement": {"days": 1, "span_days": 0}}
    )
    assert not once.recommend_specialist
    repeated = StateEvaluator().evaluate(
        analysis, [], "normal", recurrence={"maternal_estrangement": {"days": 3, "span_days": 5}}
    )
    assert repeated.specialist_reasons == ["persistent_maternal_estrangement"]


async def test_screening_result_feeds_chat_state(client, auth):
    """Проверяет использование актуального результата шкалы при выборе маршрута чата."""
    result = await complete(client, auth, "EPDS", epds_answers(12))
    assert result["result"]["recommended_action"] == "professional_assessment"
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Привет")
    steps, report = stages(response)
    assert steps["state_evaluation"]["facts"]["specialist_reasons"] == ["screening_followup:EPDS"]
    assert report["result"]["route"] == "recommend_professional"
    assert "специалист" in response.json()["assistant_message"]["content"]


async def test_questionnaire_safety_item_feeds_chat_state(client, auth):
    """Проверяет передачу safety-флага завершённой шкалы в решение чата."""
    answers = minimum_answers("EPDS") | {"10": 1}  # «Иногда»
    result = await complete(client, auth, "EPDS", answers)
    assert result["safety_flags"] == ["10"] and "112" in result["safety_guidance"]
    cid = await conversation(client, auth)
    steps, _ = stages(await send(client, auth, cid, "Привет"))
    assert steps["state_evaluation"]["facts"]["specialist_reasons"] == [
        "screening_safety_item:EPDS"
    ]


# ---------------------------------------------------------------- screening schedule
async def backdate(sessions, code_days):
    """Сдвигает даты завершения и расписания для проверки интервалов повторной оценки."""
    async with sessions() as s, s.begin():
        for attempt in (await s.scalars(select(QuestionnaireAttempt))).all():
            days = code_days.get(attempt.questionnaire_snapshot["code"])
            if days is not None:
                attempt.completed_at = utcnow() - timedelta(days=days)
        await s.execute(update(AssessmentSchedule).values(next_due_at=None))


async def offers(sessions, config):
    """Возвращает доступные шкалы и основания их предложения на фиксированное время."""
    async with sessions() as s:
        actions = await ScreeningScheduler(config).next_actions(s, ANNA)
    return {a["code"]: a["reason"] for a in actions if a["type"] == "questionnaire"}


async def test_no_periodic_reoffer_without_reason(client, auth, sessions, config):
    """Проверяет отсутствие автоматического повтора шкал без нового основания."""
    await complete(client, auth, "EPDS", epds_answers(3))
    await complete(client, auth, "GAD2", {"1": 0, "2": 0})
    await backdate(sessions, {"EPDS": 60, "GAD2": 60})
    assert await offers(sessions, config) == {}


async def test_window_and_elevated_recheck(client, auth, sessions, config):
    """Проверяет предложения по перинатальному окну и повтор после повышенного результата."""
    await complete(client, auth, "EPDS", epds_answers(12))
    await complete(client, auth, "GAD2", {"1": 0, "2": 0})
    await backdate(sessions, {"EPDS": 5, "GAD2": 5})
    assert await offers(sessions, config) == {}
    await backdate(sessions, {"EPDS": 15, "GAD2": 15})
    assert await offers(sessions, config) == {"EPDS": "recheck_after_elevated"}
    due = (utcnow() + timedelta(weeks=10)).date().isoformat()  # 30-я неделя
    await client.patch(P + "/settings", headers=auth["anna"], json={"estimated_due_date": due})
    await backdate(sessions, {"EPDS": 40, "GAD2": 40})
    assert await offers(sessions, config) == {"EPDS": "late_pregnancy", "GAD2": "late_pregnancy"}
    await backdate(sessions, {"EPDS": 15, "GAD2": 15})  # уже пройдены внутри окна
    assert (await offers(sessions, config)).get("GAD2") is None


async def test_gad7_is_offered_once_per_gad2_trigger(client, auth, sessions, config):
    """Проверяет одно предложение GAD-7 на основание, полученное после GAD-2."""
    await complete(client, auth, "GAD2", {"1": 2, "2": 1})
    assert (await offers(sessions, config)).get("GAD7") == "gad2_followup_or_professional"
    await complete(client, auth, "GAD7", dict.fromkeys(map(str, range(1, 8)), 0))
    await backdate(sessions, {"GAD7": 40})
    assert "GAD7" not in await offers(sessions, config)


async def test_next_actions_is_read_only(client, auth, sessions):
    """Проверяет, что чтение следующих действий не создаёт записей расписания."""
    assert (await client.get(P + "/next-actions", headers=auth["anna"])).status_code == 200
    async with sessions() as s:
        assert await s.scalar(select(func.count()).select_from(AssessmentSchedule)) == 0


async def test_chat_offer_is_recorded_once(client, auth, app, sessions):
    """Проверяет фиксацию предложения шкалы и его отсутствие в следующем ответе."""

    class Offers:
        async def generate(self, messages, repair=False):
            """Проверяет наличие предложения шкалы в первом запросе к генератору."""
            assert "необязательный опросник" in messages[0]["content"]
            return "Спасибо, что написала. Если захочешь, в приложении есть короткий опросник."

    app.state.llm = Offers()
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Привет")

    class NoOffer:
        async def generate(self, messages, repair=False):
            """Проверяет отсутствие повторного предложения шкалы в запросе к генератору."""
            assert "необязательный опросник" not in messages[0]["content"]
            return "Я рядом."

    # Обе первичные шкалы уже предложены по одной за ответ; третьего предложения нет.
    app.state.llm = Offers()
    await send(client, auth, cid, "Ещё кое-что")
    app.state.llm = NoOffer()
    await send(client, auth, cid, "И ещё")


# ---------------------------------------------------------------- proactive pause ownership
async def test_user_pause_survives_chat_but_automatic_pause_is_lifted(client, auth, sessions):
    """Проверяет сохранение ручной паузы и снятие автоматической после ответа пользователя."""
    cid = await conversation(client, auth)
    until = (utcnow() + timedelta(days=20)).isoformat()
    await client.patch(
        P + "/settings", headers=auth["anna"], json={"proactive_paused_until": until}
    )
    await send(client, auth, cid, "Привет")
    settings = (await client.get(P + "/settings", headers=auth["anna"])).json()
    assert settings["proactive_paused_until"] and settings["proactive_pause_reason"] == "user"
    async with sessions() as s, s.begin():
        row = await s.get(UserSettings, ANNA)
        row.proactive_pause_reason = "unanswered"
    await send(client, auth, cid, "Я вернулась")
    settings = (await client.get(P + "/settings", headers=auth["anna"])).json()
    assert settings["proactive_paused_until"] is None and settings["proactive_pause_reason"] is None


# ---------------------------------------------------------------- reads are not blocked
@pytest.mark.postgres
@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="PostgreSQL row locks require TEST_DATABASE_URL",
)
async def test_reads_do_not_wait_for_user_lock(client, auth, sessions):
    """Проверяет, что маршруты чтения используют сессии без блокировки профиля."""
    cid = await conversation(client, auth)
    async with sessions() as s, s.begin():
        # Имитация идущей генерации ответа: строка пользователя заблокирована.
        await s.scalar(select(UserSettings).where(UserSettings.user_id == ANNA).with_for_update())
        for path in (f"/conversations/{cid}/messages", "/settings", "/next-actions", "/trends"):
            response = await asyncio.wait_for(client.get(P + path, headers=auth["anna"]), timeout=5)
            assert response.status_code == 200

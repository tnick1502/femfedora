"""Regressions for stage-aware scales, bounded state and conservative JEV gating."""

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from emo_core.core.auth import DEMO_USERS
from emo_core.core.exceptions import InvalidMaterial, ProviderUnavailable
from emo_core.db.base import utcnow
from emo_core.db.models import AssessmentSchedule, MessageSignal, Questionnaire
from emo_core.db.models import TestAttempt as Attempt
from emo_core.providers.jev_provider import TypeSafeJevProvider
from emo_core.schemas.emotion import CATEGORIES, EXTRA_SIGNALS
from emo_core.services.questionnaire_materials import validate_material
from emo_core.services.screening_engine import result_report, score_answers
from emo_core.services.screening_scheduler import ScreeningScheduler
from tests.conftest import conversation, send
from tests.test_clinical_routing import complete, offers
from tests.test_questionnaire_workflow import MATERIALS, minimum_answers, start

P = "/api/v1/emo"
ANNA = DEMO_USERS["anna"]
FROZEN_REGISTRY = json.loads(
    (
        Path(__file__).resolve().parents[1] / "alembic/data/questionnaires_20261110_v1.json"
    ).read_text()
)


def nouls(**values):
    return SimpleNamespace(nouls={k: SimpleNamespace(noul=v) for k, v in values.items()})


def full_response():
    return SimpleNamespace(
        nouls={k: SimpleNamespace(noul=0.1) for k in (*CATEGORIES, *EXTRA_SIGNALS)},
        choices={"intent": SimpleNamespace(choice="support")},
    )


async def test_gate_skips_greeting_without_fabricated_negative_markers(config):
    provider, calls = TypeSafeJevProvider(config), []

    async def ask(state, questions):
        calls.append(set(questions))
        return nouls(clinical_relevance=0.01)

    provider.ask = ask
    result = await provider.analyze("Привет!")
    assert calls == [{"clinical_relevance"}]
    assert result.available and result.scope == "irrelevant" and result.markers == {}
    assert result.input_calls == 1


@pytest.mark.parametrize(
    "text", ["Мне тревожно", "да", "опять", "...", "Стало лучше", "asdf asdf asdf"]
)
async def test_gate_cannot_skip_symptoms_or_ambiguous_continuation(config, text):
    provider, calls = TypeSafeJevProvider(config), []

    async def ask(state, questions):
        calls.append(questions)
        return full_response()

    provider.ask = ask
    result = await provider.analyze(text)
    assert len(calls) == 1 and result.scope == "full" and len(result.markers) == 24


@pytest.mark.parametrize("kind", ["timeout", "nan", "missing", "uncertain"])
async def test_failed_or_uncertain_gate_runs_full_classification(config, kind):
    provider, calls = TypeSafeJevProvider(config), []

    async def ask(state, questions):
        calls.append(questions)
        if len(calls) > 1:
            return full_response()
        if kind == "timeout":
            raise ProviderUnavailable("jev_unavailable")
        return nouls(
            **(
                {}
                if kind == "missing"
                else {"clinical_relevance": float("nan") if kind == "nan" else 0.2}
            )
        )

    provider.ask = ask
    result = await provider.analyze("Привет")
    assert result.available and result.scope == "full" and result.input_calls == 2


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("preconception", {"PHQ9", "GAD7"}),
        ("pregnant", {"PHQ9", "GAD7", "EPDS"}),
        ("postpartum", {"PHQ9", "GAD7", "EPDS", "CITYBITS"}),
        ("after_loss", {"PHQ9", "GAD7"}),
    ],
)
async def test_stage_catalog_and_plan(client, auth, stage, expected):
    payload = {"reproductive_stage": stage}
    if stage == "postpartum":
        payload["birth_date"] = (utcnow() - timedelta(days=50)).isoformat()
    response = await client.patch(P + "/settings", headers=auth["anna"], json=payload)
    assert response.status_code == 200, response.text
    catalog = (await client.get(P + "/questionnaires", headers=auth["anna"])).json()
    assert {q["code"] for q in catalog["items"]} == expected
    assert "GAD2" not in str(catalog)
    assert catalog["unavailable"] == []
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert {r["code"] for r in state["assessments"]} == {"PHQ9", "GAD7", "EPDS", "CITYBITS"}
    assert {r["code"] for r in state["assessments"] if r["status"] == "due"} == expected


async def test_stage_change_clears_previous_calendar_and_blocks_wrong_scale(client, auth, sessions):
    await client.patch(P + "/settings", headers=auth["anna"], json={"gestational_week": 28})
    async with sessions() as s:
        epds_id = str(await s.scalar(select(Questionnaire.id).where(Questionnaire.code == "EPDS")))
    changed = (
        await client.patch(
            P + "/settings", headers=auth["anna"], json={"reproductive_stage": "preconception"}
        )
    ).json()
    assert changed["gestational_week"] is None and changed["estimated_due_date"] is None
    assert (
        await client.post(
            P + "/questionnaire-attempts", headers=auth["anna"], json={"questionnaire_id": epds_id}
        )
    ).status_code == 409


def city_minimum():
    return {
        str(n): 1 if n in (1, 2, 27, 28, 29) else 3 if n in (25, 26) else 0 for n in range(1, 30)
    }


def test_citybits_exact_total_subscales_and_no_invented_sum_cutoff():
    q = Questionnaire(**MATERIALS["CITYBITS"])
    minimum = city_minimum()
    assert score_answers(q, minimum, True) == (0, [])
    extra = {**minimum, "1": 0, "2": 0, "23": 3, "24": 3, "26": 2, "29": 0}
    assert score_answers(q, extra, True)[0] == 0
    maximum = {**extra, **{str(n): 3 for n in range(3, 23)}}
    score, flags = score_answers(q, maximum, True)
    assert score == 60 and flags == ["18"]
    report = result_report(q, maximum, {}, score, flags)
    assert report["subscales"] == {
        "reexperiencing": 15,
        "avoidance": 6,
        "negative_cognitions_mood": 21,
        "hyperarousal": 18,
        "birth_related": 30,
        "general": 30,
        "dissociation": 6,
    }
    assert (
        report["interpretation"] is None
        and report["recommended_action"] == "immediate_safety_assessment"
    )
    assert report["review_indicators"]["possible_other_cause"]
    assert "diagnosis" not in report
    assert (
        result_report(q, {**minimum, "28": 0}, {}, 0, [])["recommended_action"]
        == "professional_assessment"
    )


async def test_citybits_draft_safety_and_completion_roundtrip(client, auth):
    a = await start(client, auth, "CITYBITS")
    path = P + "/questionnaire-attempts/" + a["id"]
    draft = (
        await client.patch(
            path, headers=auth["anna"], json={"expected_revision": 0, "answers": {"18": 1}}
        )
    ).json()
    assert draft["total_score"] is None and draft["safety_flags"] == ["18"]
    assert "безрассудное" in draft["safety_guidance"]
    assert "мысли о причинении" not in draft["safety_guidance"].lower()
    completed = await client.post(
        path + "/complete",
        headers=auth["anna"],
        json={"expected_revision": 1, "answers": city_minimum()},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["total_score"] == 0 and completed.json()["safety_flags"] == ["18"]
    assert completed.json()["result"]["item_scores"]["25"] is None


async def test_citybits_waits_for_postpartum_window(client, auth, sessions, config):
    await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"birth_date": (utcnow() - timedelta(days=10)).isoformat()},
    )
    async with sessions() as s:
        row = next(
            r for r in await ScreeningScheduler(config).plan(s, ANNA) if r["code"] == "CITYBITS"
        )
    assert row["status"] == "waiting_window"
    assert "CITYBITS" not in await offers(sessions, config)


async def test_chat_exact_offer_is_recorded_without_model_cooperation(client, auth, app, sessions):
    class Plain:
        async def generate(self, messages, repair=False):
            return "Привет! Что хочешь обсудить?"

    app.state.llm = Plain()
    cid = await conversation(client, auth)
    first = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "PHQ-9" in first["content"]
    async with sessions() as s:
        assert await s.scalar(select(func.count()).select_from(AssessmentSchedule)) == 1
    second = (await send(client, auth, cid, "Ещё вопрос")).json()["assistant_message"]
    assert "необязательный опросник" not in second["content"]
    assert second["metadata"]["state"]["screening_offer"] is None


async def test_draft_and_decline_in_state(client, auth):
    a = await start(client, auth, "PHQ9")
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    row = next(r for r in state["assessments"] if r["code"] == "PHQ9")
    assert row["status"] == "in_progress" and row["attempt_id"] == a["id"]
    await client.post(
        P + "/assessment-schedule/PHQ9", headers=auth["anna"], json={"action": "declined"}
    )
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert next(r for r in state["assessments"] if r["code"] == "PHQ9")["status"] == "deferred"


async def test_current_worsening_is_not_overridden_by_low_screen(client, auth, app):
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))

    class Spy:
        prompt = None

        async def generate(self, messages, repair=False):
            self.prompt = messages
            return "Что изменилось в самочувствии?"

    spy = Spy()
    app.state.llm = spy
    cid = await conversation(client, auth)
    result = (await send(client, auth, cid, "Стало хуже, тревожно")).json()["assistant_message"]
    state = result["metadata"]["state"]
    assert (
        state["change"] == "worsening"
        and "current_worsening_despite_low_screen" in state["context_conflicts"]
    )
    context = json.loads(spy.prompt[1]["content"].split("\n", 1)[1])
    assert "patient_state" in context and "screenings" not in context
    assert context["patient_state"]["short"]["current"]["self_reported_worsening"] == 1
    assert context["patient_state"]["long"]["scales"][0]["score"] == 0
    assert "не опровергает" in spy.prompt[0]["content"]


async def test_short_context_expires_and_is_tenant_scoped(client, auth, sessions):
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно")
    before = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert "anxiety" in {r["code"] for r in before["short"]["recent"]}
    other = (await client.get(P + "/state", headers=auth["maria"])).json()["patient_context"]
    assert other["short"]["recent"] == []
    async with sessions() as s, s.begin():
        for row in (await s.scalars(select(MessageSignal))).all():
            row.observed_at = utcnow() - timedelta(days=4)
    after = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert after["short"]["recent"] == []
    assert (await client.get(P + "/complaints", headers=auth["anna"])).json()["items"]


async def test_scale_delta_never_crosses_episodes(client, auth, sessions):
    for value in (0, 1):
        await complete(client, auth, "GAD7", dict.fromkeys(map(str, range(1, 8)), value))
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert state["long"]["scales"][0]["delta"] == 7
    async with sessions() as s, s.begin():
        latest = await s.scalar(select(Attempt).order_by(Attempt.completed_at.desc()).limit(1))
        latest.pregnancy_id = uuid4()
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert "delta" not in state["long"]["scales"][0]


async def test_greeting_preserves_persistent_complaint_referral(client, auth, sessions):
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
        for row, days in zip(rows, (17, 8, 1), strict=True):
            row.observed_at = utcnow() - timedelta(days=days)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "persistent_complaint" in answer["metadata"]["state"]["specialist_reasons"]
    assert answer["metadata"]["state"]["analysis_scope"] == "irrelevant"


async def test_safety_draft_reaches_chat_before_completion(client, auth):
    a = await start(client, auth, "PHQ9")
    response = await client.patch(
        P + "/questionnaire-attempts/" + a["id"],
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": {"9": 1}},
    )
    assert response.status_code == 200
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "screening_safety_item:PHQ9" in answer["metadata"]["state"]["specialist_reasons"]
    assert answer["metadata"]["state"]["referral_required"]
    assert "опросник PHQ" not in answer["content"]


async def test_many_repeats_of_one_scale_do_not_evict_other_recent_result(client, auth, sessions):
    phq = minimum_answers("PHQ9") | {"1": 3, "2": 3, "3": 3, "4": 1}
    await complete(client, auth, "PHQ9", phq)
    # A burst of one scale must not hide a still-current result from another scale.
    gad = await complete(client, auth, "GAD7", minimum_answers("GAD7"))
    async with sessions() as s, s.begin():
        template = await s.get(Attempt, __import__("uuid").UUID(gad["id"]))
        for n in range(25):
            s.add(
                Attempt(
                    user_id=ANNA,
                    questionnaire_id=template.questionnaire_id,
                    answers=template.answers,
                    auxiliary_answers={},
                    safety_flags=[],
                    context=template.context,
                    questionnaire_snapshot=template.questionnaire_snapshot,
                    result=template.result,
                    total_score=0,
                    status="completed",
                    completed_at=utcnow() + timedelta(microseconds=n),
                )
            )
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "screening_followup:PHQ9" in answer["metadata"]["state"]["specialist_reasons"]
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert len(state["long"]["scales"]) == 2


def test_practical_help_respects_physical_and_violence_constraints(config):
    from emo_core.schemas.emotion import SupportState
    from emo_core.services.support_protocol_selector import SupportProtocolSelector

    selector = SupportProtocolSelector(config)
    state = SupportState(
        support_protocol="low_mood_support",
        current_signals=["physical_discomfort", "possible_violence_exposure"],
    )
    prompt = selector.system_prompt(state, None, None)
    assert "не советуй прогулки" in prompt.lower()
    assert "не советуй конфронтацию" in prompt.lower()
    assert "не назначай" in prompt.lower() or "назначать, менять или отменять" in prompt.lower()
    assert selector.select(state) == (None, None)  # No invented approval.


async def test_support_continuation_uses_recent_topic_without_fabricating_current_symptoms(
    client, auth, app
):
    class Spy:
        prompt = None

        async def generate(self, messages, repair=False):
            self.prompt = messages
            return "Хочешь продолжить разговор о том, что беспокоило?"

    spy = Spy()
    app.state.llm = spy
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно")
    continued = (await send(client, auth, cid, "Хочу поговорить")).json()["assistant_message"]
    state = continued["metadata"]["state"]
    assert state["primary_support_need"] == "anxiety"
    assert state["support_basis"] == "recent_observation_needs_confirmation"
    assert "anxiety" not in state["current_signals"]
    assert "сначала коротко уточни" in spy.prompt[0]["content"]
    assert "equal_markers_fixed_category_priority" not in str(
        continued["metadata"]["decision_report"]
    )
    greeting = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert greeting["metadata"]["state"]["primary_support_need"] == "general"
    assert greeting["metadata"]["state"]["analysis_scope"] == "irrelevant"


async def test_support_request_can_use_fresh_screen_but_not_claim_current_low_mood(client, auth):
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9") | {"1": 3, "2": 3, "3": 3, "4": 1})
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Хочу поговорить")).json()["assistant_message"]
    state = answer["metadata"]["state"]
    assert state["primary_support_need"] == "low_mood"
    assert state["support_basis"] == "screening_result_needs_confirmation"
    assert "low_mood" not in state["current_signals"]
    assert state["recommend_specialist"]
    assert "screening_trends" not in state and "screening_status" not in state


async def test_assessment_history_is_bounded_and_shared_by_chat(client, auth, app, sessions):
    from sqlalchemy import event

    await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))
    await start(client, auth, "GAD7")
    cid = await conversation(client, auth)
    statements = []
    engine = sessions.kw["bind"].sync_engine

    def capture(conn, cursor, statement, params, context, executemany):
        if "row_number() OVER" in statement and "test_attempts" in statement:
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = await send(client, auth, cid, "Привет")
        assert response.status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(statements) == 1


async def test_old_phq9_draft_keeps_its_verified_material_after_upgrade(client, auth, sessions):
    await client.patch(P + "/settings", headers=auth["anna"], json={})
    old = copy.deepcopy(next(q for q in FROZEN_REGISTRY if q["code"] == "PHQ9"))
    old["enabled"] = False
    assert validate_material(old)
    async with sessions() as session, session.begin():
        questionnaire = Questionnaire(**old)
        session.add(questionnaire)
        await session.flush()
        attempt = Attempt(
            user_id=ANNA,
            questionnaire_id=questionnaire.id,
            answers={"1": 0},
            auxiliary_answers={},
            safety_flags=[],
            status="in_progress",
            context={"perinatal_stage": "unknown"},
        )
        session.add(attempt)
        await session.flush()
        attempt_id = str(attempt.id)
    response = await client.post(
        P + "/questionnaire-attempts/" + attempt_id + "/complete",
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": minimum_answers("PHQ9")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["questionnaire_version"] == old["version"]
    assert response.json()["total_score"] == 0
    changed = copy.deepcopy(old)
    changed["definition"]["items"][0]["text"] += " altered"
    with pytest.raises(InvalidMaterial):
        validate_material(changed)


async def test_previous_birth_result_and_draft_are_not_current_even_without_episode_id(
    client, auth, sessions
):
    new_birth = utcnow() - timedelta(days=60)
    await client.patch(
        P + "/settings", headers=auth["anna"], json={"birth_date": new_birth.isoformat()}
    )
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9") | {"1": 3, "2": 3, "3": 3, "4": 1})
    draft = await start(client, auth, "GAD7")
    async with sessions() as session, session.begin():
        for attempt in (
            await session.scalars(select(Attempt).where(Attempt.user_id == ANNA))
        ).all():
            attempt.context = {
                **attempt.context,
                "birth_date": (new_birth - timedelta(days=400)).isoformat(),
            }
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert state["long"]["scales"][0]["other_episode"]
    assert next(r for r in state["assessments"] if r["code"] == "GAD7")["status"] == "due"
    assert draft["id"] not in str(state["assessments"])
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "screening_followup:PHQ9" not in answer["metadata"]["state"]["specialist_reasons"]

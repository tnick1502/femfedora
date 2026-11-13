"""Boundary regressions from the independent state/screening audit."""

import copy
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from emo_core.db.base import utcnow
from emo_core.db.models import Checkin, Questionnaire
from emo_core.db.models import TestAttempt as Attempt
from emo_core.providers.jev_provider import TypeSafeJevProvider
from emo_core.schemas.emotion import CATEGORIES, EXTRA_SIGNALS
from emo_core.services.context_manager import ChatContextManager, ContextBudget, estimate_tokens
from emo_core.services.patient_state import prompt_patient_state
from tests.conftest import conversation, send, synthetic_material
from tests.test_clinical_routing import complete
from tests.test_patient_state_v2 import ANNA, full_response, nouls
from tests.test_questionnaire_workflow import minimum_answers, start

P = "/api/v1/emo"


async def patient(client, auth):
    return (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]


async def test_disabled_passr_cannot_start_even_if_database_enabled(client, auth, sessions):
    await client.patch(
        P + "/settings", headers=auth["anna"], json={"reproductive_stage": "pregnant"}
    )
    data = synthetic_material()
    data["code"] = "PASSR"
    async with sessions() as s, s.begin():
        q = Questionnaire(**data)
        s.add(q)
        await s.flush()
        qid = str(q.id)
    response = await client.post(
        P + "/questionnaire-attempts", headers=auth["anna"], json={"questionnaire_id": qid}
    )
    assert response.status_code == 409
    catalog = (await client.get(P + "/questionnaires", headers=auth["anna"])).json()
    assert "PASSR" not in str(catalog)
    assert "PASSR" not in str((await patient(client, auth))["assessments"])


async def test_null_episode_does_not_restore_previous_pregnancy(client, auth):
    await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"pregnancy_id": str(uuid4()), "gestational_week": 12},
    )
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно")
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))
    await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"pregnancy_id": None, "reproductive_stage": "preconception"},
    )
    data = await patient(client, auth)
    assert data["short"]["recent"] == []
    assert data["long"]["patterns"] == {}
    assert data["long"]["scales"][0]["other_episode"]


async def test_calendar_only_new_pregnancy_clears_previous_birth(client, auth):
    await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"birth_date": (utcnow() - timedelta(days=90)).isoformat()},
    )
    response = await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"pregnancy_id": str(uuid4()), "gestational_week": 8},
    )
    assert response.status_code == 200
    assert response.json()["birth_date"] is None
    assert (await patient(client, auth))["assessments"][0]["stage"] == "pregnant"


async def test_checkin_uses_previous_distinct_day_and_keeps_missing_sleep(client, auth, sessions):
    await client.patch(P + "/settings", headers=auth["anna"], json={})
    now = utcnow()
    async with sessions() as s, s.begin():
        for at, mood, sleep in (
            (now - timedelta(days=2), 2, 3),
            (now - timedelta(minutes=2), 4, 5),
            (now, 6, None),
        ):
            s.add(
                Checkin(
                    user_id=ANNA,
                    created_at=at,
                    mood_score=mood,
                    anxiety_score=5,
                    sleep_quality=sleep,
                )
            )
    checkin = (await patient(client, auth))["long"]["checkin"]
    assert checkin["delta"] == {"mood": 4, "anxiety": 0}
    assert checkin["previous_at"] == (now - timedelta(days=2)).date().isoformat()
    trends = (await client.get(P + "/trends", headers=auth["anna"])).json()["checkins"]
    assert trends["deltas"] == {"mood_score": 4, "anxiety_score": 0, "sleep_quality": None}


async def test_old_draft_new_safety_answer_survives_newer_empty_draft(client, auth, sessions):
    old = await start(client, auth, "PHQ9")
    async with sessions() as s, s.begin():
        row = await s.get(Attempt, UUID(old["id"]))
        row.started_at = utcnow() - timedelta(days=60)
    await client.patch(
        P + "/questionnaire-attempts/" + old["id"],
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": {"9": 1}},
    )
    await start(client, auth, "PHQ9")
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "screening_safety_item:PHQ9" in answer["metadata"]["state"]["specialist_reasons"]
    assert answer["metadata"]["state"]["referral_required"]


async def test_new_safety_item_bypasses_previous_routine_referral_cooldown(client, auth):
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9") | {"1": 3, "2": 3, "3": 3, "4": 1})
    cid = await conversation(client, auth)
    first = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert first["metadata"]["state"]["referral_required"]
    draft = await start(client, auth, "PHQ9")
    await client.patch(
        P + "/questionnaire-attempts/" + draft["id"],
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": {"9": 1}},
    )
    second = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert second["metadata"]["state"]["referral_required"]
    assert second["metadata"]["state"]["screening_offer"] is None


async def test_full_classifier_gets_previous_question_for_short_answer(config):
    provider = TypeSafeJevProvider(config)
    calls = []

    async def ask(state, questions):
        calls.append((state, questions))
        return full_response()

    provider.ask = ask
    prior = [{"role": "assistant", "content": "Из-за этого трудно работать?"}]
    await provider.analyze_with_context("Да", prior)
    assert calls[0][0] == {"message": "Да", "previous_turns": prior}
    assert set(calls[0][1]) == set(CATEGORIES + EXTRA_SIGNALS) | {"intent"}


async def test_boolean_gate_answer_does_not_skip_classification(config):
    provider = TypeSafeJevProvider(config)
    calls = []

    async def ask(state, questions):
        calls.append(questions)
        return nouls(clinical_relevance=False) if len(calls) == 1 else full_response()

    provider.ask = ask
    assert (await provider.analyze("Привет")).scope == "full"
    assert len(calls) == 2


def test_summarization_receives_tail_disclosure_after_long_message_compression():
    history = [
        {"role": "user", "content": "обычный рассказ " * 300 + "DISCLOSURE_AT_END", "index": 0},
        {"role": "assistant", "content": "ответ", "index": 1},
        {"role": "user", "content": "продолжение", "index": 2},
    ]
    manager = ChatContextManager(ContextBudget(input_target_tokens=512, keep_recent_messages=2))
    plan = manager.fit("Правила", {"context": "данные " * 200}, history)
    assert plan.to_summarize
    assert plan.to_summarize[0]["content"].endswith("DISCLOSURE_AT_END")


async def test_compact_state_keeps_clinical_information(client, auth):
    import json

    await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))
    full = await patient(client, auth)
    original = copy.deepcopy(full)
    compact = prompt_patient_state(full)
    assert full == original
    assert compact["long"] == full["long"]
    assert compact["short"]["current"] == full["short"]["current"]
    assert len(compact["assessments"]) == len(full["assessments"])
    assert estimate_tokens(json.dumps(compact)) < estimate_tokens(json.dumps(full)) * 0.8


async def test_positive_safety_item_suppresses_routine_next_actions(client, auth):
    draft = await start(client, auth, "PHQ9")
    await client.patch(
        P + "/questionnaire-attempts/" + draft["id"],
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": {"9": 1}},
    )
    actions = (await client.get(P + "/next-actions", headers=auth["anna"])).json()["items"]
    assert actions == [
        {"type": "professional_support", "reason": "screening_safety_priority", "optional": False}
    ]


async def test_phq_functional_difficulty_is_not_hidden_by_low_score(client, auth):
    draft = await start(client, auth, "PHQ9")
    result = await client.post(
        P + "/questionnaire-attempts/" + draft["id"] + "/complete",
        headers=auth["anna"],
        json={
            "expected_revision": 0,
            "answers": minimum_answers("PHQ9") | {"1": 1},
            "auxiliary_answers": {"difficulty": 3},
        },
    )
    assert result.status_code == 200
    assert result.json()["total_score"] == 1
    assert result.json()["result"]["recommended_action"] == "professional_assessment"
    assert result.json()["result"]["review_indicators"]["functional_impact"]


async def test_explicit_memory_correction_survives_refresh(client, auth, app):
    class Spy:
        prompt = None

        async def generate(self, messages, repair=False):
            self.prompt = messages
            return "Как ты сейчас?"

    spy = Spy()
    app.state.llm = spy
    cid = await conversation(client, auth)
    await client.patch(
        P + "/memory",
        headers=auth["anna"],
        json={"summary": "Прогулки запрещены моим врачом", "preferences": {}},
    )
    # Explicit change triggers automatic memory refresh immediately.
    await send(client, auth, cid, "Стало лучше")
    await send(client, auth, cid, "Хочу поговорить")
    state = (await client.get(P + "/state", headers=auth["anna"])).json()
    assert "Прогулки запрещены" in state["summary"]
    assert "Прогулки запрещены" in spy.prompt[1]["content"]
    await client.delete(P + "/memory", headers=auth["anna"])
    other = await conversation(client, auth)
    await send(client, auth, other, "Стало лучше")
    state = (await client.get(P + "/state", headers=auth["anna"])).json()
    assert "Прогулки запрещены" not in state["summary"]


async def test_old_draft_is_not_offered_after_newer_completion(client, auth):
    await start(client, auth, "PHQ9")
    await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))
    row = next(r for r in (await patient(client, auth))["assessments"] if r["code"] == "PHQ9")
    assert row["status"] == "up_to_date"
    actions = (await client.get(P + "/next-actions", headers=auth["anna"])).json()["items"]
    assert not any(a.get("code") == "PHQ9" for a in actions)


async def test_offer_is_cancelled_when_test_completes_during_generation(client, auth, app):
    class CompletesTest:
        async def generate(self, messages, repair=False):
            await complete(client, auth, "PHQ9", minimum_answers("PHQ9"))
            return "Привет! О чём хочешь поговорить?"

    app.state.llm = CompletesTest()
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "необязательный опросник" not in answer["content"]
    assert answer["metadata"]["state"]["screening_offer"] is None


@pytest.mark.parametrize("change", ["decline", "stage", "safety"])
async def test_offer_is_cancelled_when_context_changes_during_generation(client, auth, app, change):
    class ChangesContext:
        async def generate(self, messages, repair=False):
            if change == "decline":
                await client.post(
                    P + "/assessment-schedule/PHQ9",
                    headers=auth["anna"],
                    json={"action": "declined"},
                )
            elif change == "stage":
                await client.patch(
                    P + "/settings",
                    headers=auth["anna"],
                    json={"reproductive_stage": "preconception"},
                )
            else:
                draft = await start(client, auth, "PHQ9")
                await client.patch(
                    P + "/questionnaire-attempts/" + draft["id"],
                    headers=auth["anna"],
                    json={"expected_revision": 0, "answers": {"9": 1}},
                )
            return "Привет! О чём хочешь поговорить?"

    app.state.llm = ChangesContext()
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert "необязательный опросник" not in answer["content"]
    assert answer["metadata"]["state"]["screening_offer"] is None

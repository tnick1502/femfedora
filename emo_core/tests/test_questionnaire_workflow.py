"""Real published keys and API contracts; no LLM or invented questionnaire text."""

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from emo_core.core.exceptions import InvalidMaterial
from emo_core.db.models import Questionnaire, SafetyEvent
from emo_core.db.models import TestAttempt as QuestionnaireAttempt
from emo_core.services.questionnaire_materials import material_hash, validate_material
from emo_core.services.screening_engine import result_report, score_answers
from tests.conftest import REGISTRY, synthetic_material

P = "/api/v1/emo"
MATERIALS = {r["code"]: r for r in REGISTRY}


def minimum_answers(code):
    """Возвращает индексы минимально оцениваемых вариантов встроенной редакции шкалы."""
    if code == "EPDS":
        return dict(zip(map(str, range(1, 11)), [0, 0, 3, 0, 3, 3, 3, 3, 3, 3], strict=True))
    return dict.fromkeys(
        (str(i + 1) for i in range(len(MATERIALS[code]["definition"]["items"]))), 0
    )


async def catalog(client, auth, code):
    """Загружает каталог доступных шкал через API тестового профиля."""
    response = await client.get(P + "/questionnaires", headers=auth["anna"])
    assert response.status_code == 200
    return next(q for q in response.json()["items"] if q["code"] == code)


async def start(client, auth, code="GAD2", request_id=None):
    """Начинает прохождение с новым UUID идемпотентности и возвращает его представление."""
    q = await catalog(client, auth, code)
    body = {"questionnaire_id": q["id"], "client_request_id": request_id or str(uuid4())}
    response = await client.post(P + "/questionnaire-attempts", headers=auth["anna"], json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_bundled_source_integrity_and_no_fictitious_approval():
    """Проверяет совпадение встроенных форм с данными миграции и отсутствие выдуманного клинического одобрения."""
    root = Path(__file__).resolve().parents[1]
    for data in REGISTRY:
        assert validate_material(data) == data
        assert data["definition"]["clinical_review_status"] == "not_performed"
        assert data["definition"]["source_review_status"] == "verified"
        assert data["definition"]["attribution"]
        for ref in data["definition"]["references"]:
            if ref.get("source_file"):
                assert (
                    hashlib.sha256((root / ref["source_file"]).read_bytes()).hexdigest()
                    == ref["sha256"]
                )
    frozen = json.loads((root / "alembic/data/questionnaires_20261110_v1.json").read_text())
    assert {material_hash(q) for q in frozen} == {material_hash(q) for q in REGISTRY}


def test_epds_all_published_response_keys_and_worked_example():
    """Проверяет прямые и обратные ключи EPDS, пределы суммы и опубликованный пример."""
    q = Questionnaire(**MATERIALS["EPDS"])
    for item in q.definition["items"]:
        assert item["score_map"] == (
            [0, 1, 2, 3] if item["id"] in ("1", "2", "4") else [3, 2, 1, 0]
        )
    answers = dict(zip(map(str, range(1, 11)), [1, 2, 0, 3, 1, 2, 3, 0, 1, 2], strict=True))
    assert score_answers(q, answers, True) == (18, ["10"])
    assert score_answers(q, minimum_answers("EPDS"), True) == (0, [])
    assert score_answers(q, {k: 3 - v for k, v in minimum_answers("EPDS").items()}, True) == (
        30,
        ["10"],
    )


@pytest.mark.parametrize("code,max_score", [("GAD2", 6), ("GAD7", 21), ("PHQ9", 27)])
def test_published_forward_keys_and_range(code, max_score):
    """Проверяет прямые ключи и диапазон сумм GAD-2, GAD-7 и PHQ-9."""
    q = Questionnaire(**MATERIALS[code])
    assert all(i["score_map"] == [0, 1, 2, 3] for i in q.definition["items"])
    assert score_answers(q, minimum_answers(code), True)[0] == 0
    assert score_answers(q, dict.fromkeys(minimum_answers(code), 3), True)[0] == max_score


def test_gad2_is_exact_published_gad7_subset():
    """Проверяет совпадение GAD-2 с первыми пунктами GAD-7 и периоды самооценки шкал."""
    assert MATERIALS["GAD2"]["definition"]["items"] == MATERIALS["GAD7"]["definition"]["items"][:2]
    assert MATERIALS["GAD2"]["definition"]["timeframe_days"] == 14
    assert MATERIALS["EPDS"]["definition"]["timeframe_days"] == 7


@pytest.mark.parametrize(
    "code,score,band,action",
    [
        ("GAD2", 2, "below_followup_threshold", "discuss_if_concerned"),
        ("GAD2", 3, "followup_suggested", "gad7_or_professional_assessment"),
        ("GAD7", 4, "minimal", "discuss_if_concerned"),
        ("GAD7", 5, "mild", "discuss_if_concerned"),
        ("GAD7", 9, "mild", "discuss_if_concerned"),
        ("GAD7", 10, "moderate", "professional_assessment"),
        ("GAD7", 14, "moderate", "professional_assessment"),
        ("GAD7", 15, "severe", "professional_assessment"),
        ("PHQ9", 4, "minimal", "discuss_if_concerned"),
        ("PHQ9", 5, "mild", "discuss_if_concerned"),
        ("PHQ9", 10, "moderate", "professional_assessment"),
        ("PHQ9", 15, "moderately_severe", "professional_assessment"),
        ("PHQ9", 20, "severe", "professional_assessment"),
        ("EPDS", 9, "below_followup_threshold", "discuss_if_concerned"),
        ("EPDS", 10, "followup_suggested", "professional_assessment"),
    ],
)
def test_sourced_interpretation_boundaries(code, score, band, action):
    """Проверяет граничные баллы интерпретаций и соответствующие следующие шаги."""
    q = Questionnaire(**MATERIALS[code])
    answers = minimum_answers(code)
    remaining = score
    for item in q.definition["items"]:
        value = min(3, remaining)
        answers[item["id"]] = item["score_map"].index(value)
        remaining -= value
    total, _ = score_answers(q, answers, True)
    assert total == score
    result = result_report(q, answers, {}, total, [])
    assert result["interpretation"]["code"] == band
    assert result["recommended_action"] == action
    assert result["calculated_by"] == "python" and result["models_used"] == []
    assert result["not_a_diagnosis"] is True


def test_source_verified_is_not_an_unrestricted_import_bypass():
    """Проверяет, что изменение материала нельзя одобрить одним флагом source_verified."""
    data = copy.deepcopy(MATERIALS["GAD7"])
    data["definition"]["items"][0]["text"] = "unreviewed modification"
    with pytest.raises(InvalidMaterial):
        validate_material(data)
    custom = synthetic_material()
    custom["definition"]["clinical_review_status"] = "not_performed"
    custom["definition"]["source_review_status"] = "verified"
    with pytest.raises(InvalidMaterial):
        validate_material(custom)


async def test_real_catalog_and_openapi_contract(client, auth):
    """Проверяет доступность четырёх встроенных форм и описание workflow в OpenAPI."""
    data = (await client.get(P + "/questionnaires", headers=auth["anna"])).json()
    assert {q["code"] for q in data["items"]} == set(MATERIALS)
    assert not data["unavailable"]
    q = data["items"][0]
    assert (await client.get(P + "/questionnaires/" + q["id"], headers=auth["anna"])).json() == q
    spec = (await client.get("/openapi.json")).json()
    assert spec["paths"][P + "/questionnaire-attempts/{attempt_id}"]["patch"]["requestBody"]
    assert spec["components"]["schemas"]["AttemptView"]["properties"]["result"]


async def test_start_idempotency_is_scoped_and_detects_key_reuse(client, auth):
    """Проверяет повтор старта в пределах профиля и конфликт UUID с другим телом запроса."""
    key = str(uuid4())
    a = await start(client, auth, request_id=key)
    again = await start(client, auth, request_id=key)
    assert again["id"] == a["id"] and again["revision"] == 0
    response = await client.post(
        P + "/questionnaire-attempts",
        headers=auth["anna"],
        json={
            "questionnaire_id": a["questionnaire_id"],
            "client_request_id": key,
            "answers": {"1": 2},
        },
    )
    assert response.status_code == 409
    response = await client.post(
        P + "/questionnaire-attempts",
        headers=auth["maria"],
        json={"questionnaire_id": a["questionnaire_id"], "client_request_id": key},
    )
    assert response.status_code == 200 and response.json()["id"] != a["id"]
    history = (await client.get(P + "/questionnaire-attempts", headers=auth["anna"])).json()[
        "items"
    ]
    assert len(history) == 1


async def test_resume_revision_validation_and_idempotent_completion(client, auth):
    """Проверяет продолжение черновика, ревизии, полноту ответов и повтор завершения."""
    a = await start(client, auth)
    path = P + "/questionnaire-attempts/" + a["id"]
    response = await client.patch(
        path, headers=auth["anna"], json={"expected_revision": 0, "answers": {"1": 1}}
    )
    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["total_score"] is None and draft["result"] is None
    assert draft["progress"] == {"answered": 1, "required": 2, "missing_item_ids": ["2"]}
    assert (await client.get(path, headers=auth["anna"])).json() == draft
    response = await client.patch(
        path, headers=auth["anna"], json={"expected_revision": 0, "answers": {"2": 1}}
    )
    assert response.status_code == 409 and response.json()["detail"]["code"] == "revision_conflict"
    response = await client.post(
        path + "/complete", headers=auth["anna"], json={"expected_revision": 1}
    )
    assert response.status_code == 422 and response.json()["detail"]["missing_item_ids"] == ["2"]
    body = {"expected_revision": 1, "answers": {"2": 2}}
    result = await client.post(path + "/complete", headers=auth["anna"], json=body)
    assert result.status_code == 200, result.text
    completed = result.json()
    assert completed["total_score"] == 3
    assert completed["result"]["item_scores"] == {"1": 1, "2": 2}
    assert completed["result"]["recommended_action"] == "gad7_or_professional_assessment"
    assert (
        await client.post(path + "/complete", headers=auth["anna"], json=body)
    ).json() == completed
    assert (
        await client.patch(path, headers=auth["anna"], json={"expected_revision": 2})
    ).status_code == 409
    assert (
        await client.post(
            path + "/complete",
            headers=auth["anna"],
            json={"expected_revision": 2, "answers": {"1": 3}},
        )
    ).status_code == 409
    assert (
        await client.get(P + "/questionnaire-attempts?status=in_progress", headers=auth["anna"])
    ).json()["items"] == []
    trends = (await client.get(P + "/trends", headers=auth["anna"])).json()["questionnaires"][
        "GAD2"
    ]
    assert len(trends) == 1 and trends[0]["score"] == 3


@pytest.mark.parametrize(
    "answers", [{"1": True}, {"1": "1"}, {"1": 1.0}, {"1": -1}, {"1": 4}, {"unknown": 0}]
)
async def test_strict_answer_indices(client, auth, answers):
    """Проверяет отказ от неизвестных пунктов и нецелых либо выходящих за диапазон индексов."""
    a = await start(client, auth)
    response = await client.patch(
        P + "/questionnaire-attempts/" + a["id"],
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": answers},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("code,item,positive,zero", [("EPDS", "10", 2, 3), ("PHQ9", "9", 1, 0)])
async def test_safety_independent_partial_sticky_and_deduplicated(
    client, auth, sessions, code, item, positive, zero
):
    """Проверяет немедленный safety-флаг в черновике, его сохранение и отсутствие повторных событий."""
    a = await start(client, auth, code)
    path = P + "/questionnaire-attempts/" + a["id"]
    response = await client.patch(
        path, headers=auth["anna"], json={"expected_revision": 0, "answers": {item: positive}}
    )
    assert response.status_code == 200
    assert response.json()["total_score"] is None and response.json()["safety_flags"] == [item]
    assert "сейчас" in response.json()["safety_guidance"]
    response = await client.patch(
        path, headers=auth["anna"], json={"expected_revision": 1, "answers": {item: zero}}
    )
    assert response.json()["safety_flags"] == [item] and response.json()["safety_guidance"]
    response = await client.post(
        path + "/complete",
        headers=auth["anna"],
        json={"expected_revision": 2, "answers": minimum_answers(code)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["total_score"] == 0
    assert response.json()["result"]["recommended_action"] == "immediate_safety_assessment"
    async with sessions() as s:
        assert await s.scalar(select(func.count()).select_from(SafetyEvent)) == 1


async def test_phq9_functional_item_not_in_sum(client, auth):
    """Проверяет отдельное хранение дополнительного вопроса PHQ-9 без влияния на сумму."""
    a = await start(client, auth, "PHQ9")
    response = await client.post(
        P + "/questionnaire-attempts/" + a["id"] + "/complete",
        headers=auth["anna"],
        json={
            "expected_revision": 0,
            "answers": minimum_answers("PHQ9"),
            "auxiliary_answers": {"difficulty": 3},
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_score"] == 0 and data["result"]["maximum_score"] == 27
    assert data["auxiliary_answers"] == {"difficulty": 3}
    assert data["result"]["auxiliary_in_total"] is False


async def test_snapshot_survives_registry_change_and_disabled_version(client, auth, sessions):
    """Проверяет завершение по исходному снимку после смены или отключения редакции в реестре."""
    a = await start(client, auth)
    async with sessions() as s, s.begin():
        q = await s.get(Questionnaire, UUID(a["questionnaire_id"]))
        changed = copy.deepcopy(q.definition)
        changed["items"][0]["text"] = "Not the original question"
        q.definition, q.enabled = changed, False
    response = await client.post(
        P + "/questionnaire-attempts/" + a["id"] + "/complete",
        headers=auth["anna"],
        json={"expected_revision": 0, "answers": {"1": 0, "2": 0}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["questionnaire"] == a["questionnaire"]
    assert response.json()["result"]["questionnaire_hash"] == a["questionnaire_hash"]
    assert (
        await client.post(
            P + "/questionnaire-attempts",
            headers=auth["anna"],
            json={"questionnaire_id": a["questionnaire_id"]},
        )
    ).status_code == 409


async def test_attempt_endpoints_tenant_isolation(client, auth, sessions):
    """Проверяет отсутствие доступа к чужому прохождению во всех маршрутах workflow."""
    a = await start(client, auth)
    for attempt_id in (a["id"], str(uuid4())):
        path = P + "/questionnaire-attempts/" + attempt_id
        assert (await client.get(path, headers=auth["maria"])).status_code == 404
        assert (
            await client.patch(path, headers=auth["maria"], json={"expected_revision": 0})
        ).status_code == 404
        assert (
            await client.post(
                path + "/complete", headers=auth["maria"], json={"expected_revision": 0}
            )
        ).status_code == 404
    async with sessions() as s:
        assert await s.scalar(select(func.count()).select_from(QuestionnaireAttempt)) == 1


async def test_no_duplicate_automatic_offers_or_offer_of_active_draft(client, auth):
    """Проверяет исключение дублирующих шкал и шкалы с уже начатым черновиком из предложений."""
    await start(client, auth, "EPDS")
    actions = (await client.get(P + "/next-actions", headers=auth["anna"])).json()["items"]
    offered = {a.get("code") for a in actions}
    assert "PHQ9" not in offered and "EPDS" not in offered and "GAD7" not in offered
    assert "GAD2" in offered


async def test_import_retires_old_version_without_losing_attempt(
    client, auth, sessions, monkeypatch
):
    """Проверяет отключение старой редакции при импорте новой с сохранением истории попыток."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from emo_core import cli
    from emo_core.db import session as database

    a = await start(client, auth)
    monkeypatch.setattr(database, "Session", sessions)
    monkeypatch.setattr(database, "engine", SimpleNamespace(dispose=AsyncMock()))
    await cli.execute("import-questionnaire", synthetic_material(version="approved-new-test-only"))
    async with sessions() as s:
        old = await s.get(Questionnaire, UUID(a["questionnaire_id"]))
        assert old.enabled is False
        active = list(
            (
                await s.scalars(
                    select(Questionnaire).where(
                        Questionnaire.code == "GAD2", Questionnaire.enabled.is_(True)
                    )
                )
            ).all()
        )
        assert len(active) == 1 and active[0].version == "approved-new-test-only"
    path = P + "/questionnaire-attempts/" + a["id"]
    assert (await client.get(path, headers=auth["anna"])).json()["questionnaire_hash"] == a[
        "questionnaire_hash"
    ]
    with pytest.raises(ValueError, match="неизменяема"):
        await cli.execute(
            "import-questionnaire", synthetic_material(version="approved-new-test-only")
        )

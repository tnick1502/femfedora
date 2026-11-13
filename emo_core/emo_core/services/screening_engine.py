import copy
import hashlib
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import select

from emo_core.core.exceptions import InvalidMaterial
from emo_core.db.base import utcnow
from emo_core.db.models import AssessmentSchedule, Questionnaire, TestAttempt, UserSettings
from emo_core.db.repositories import public
from emo_core.services.perinatal import perinatal_context, same_assessment_episode
from emo_core.services.policies import policy
from emo_core.services.questionnaire_materials import (
    material_hash,
    questionnaire_view,
    validate_material,
)
from emo_core.services.safety_guard import SafetyGuard
from emo_core.services.screening_scheduler import (
    SCALE_CODES,
    STAGE_SCALES,
    ScreeningScheduler,
    assessment_history,
)

# Re-exported for existing integrations / administrative imports.
__all__ = ["ScreeningEngine", "score_answers", "validate_material"]

ELEVATED_ACTIONS = (
    "immediate_safety_assessment",
    "professional_assessment",
)


async def recent_results(session, user_id, now, valid_days=30, *, history=None):
    """Возвращает свежие результаты и safety-флаги последних черновиков каждой шкалы.

    Оконная выборка ограничена одной попыткой каждого статуса на код.
    Сырые ответы модели не передаются, результат модель не вычисляет.
    """
    rows = history if history is not None else await assessment_history(session, user_id)
    user = await session.get(UserSettings, user_id)
    latest = {}
    seen = set()
    for attempt, questionnaire in rows:
        if not same_assessment_episode(attempt, user):
            continue
        code = questionnaire.code
        at = attempt.completed_at or attempt.started_at
        safety_at = (
            datetime.fromisoformat(attempt.context["safety_observed_at"])
            if attempt.context.get("safety_observed_at")
            else at
        )
        fresh_safety = (
            bool(attempt.safety_flags) and now - timedelta(days=valid_days) <= safety_at <= now
        )
        current_result = attempt.status == "completed" and code not in seen
        if attempt.status == "completed":
            seen.add(code)
        current_result = current_result and now - timedelta(days=valid_days) <= at <= now
        if not current_result and not fresh_safety:
            continue
        row = latest.setdefault(code, {"code": code, "followup": False, "safety_flags": False})
        if current_result:
            result = attempt.result or {}
            row.update(
                score=attempt.total_score,
                maximum=result.get("maximum_score"),
                at=at.isoformat(),
                band=(result.get("interpretation") or {}).get("code"),
                followup=result.get("recommended_action") in ELEVATED_ACTIONS,
            )
        if fresh_safety:
            row["safety_flags"] = True
            row["safety_at"] = max(row.get("safety_at", ""), safety_at.isoformat())
    return list(latest.values())


def validated_answers(items, answers):
    """Проверяет идентификаторы и индексы ответов, возвращает пункты по id.

    Индекс — целое число от нуля до числа вариантов минус один; bool не допускается.
    Неизвестный пункт или неверный индекс приводит к HTTP 422.
    """
    by_id = {item["id"]: item for item in items}
    if set(answers) - by_id.keys() or any(
        type(value) is not int or not 0 <= value < len(by_id[key]["options"])
        for key, value in answers.items()
        if key in by_id
    ):
        raise HTTPException(422, "Неизвестный пункт или неверный индекс варианта ответа")
    return by_id


def score_answers(questionnaire, answers, complete):
    """Возвращает сумму по ключам пунктов и список сработавших safety-пунктов.

    В черновике сумма равна None, но safety-пункты проверяются сразу. Для complete
    нужны все основные ответы, иначе HTTP 422. Дополнительные вопросы сюда не входят.
    """
    items = validated_answers(questionnaire.definition["items"], answers)
    if complete and answers.keys() != items.keys():
        raise HTTPException(
            422,
            {
                "code": "incomplete_answers",
                "message": "Для завершения ответьте на все вопросы",
                "missing_item_ids": sorted(items.keys() - answers.keys()),
            },
        )
    scores = {key: items[key]["score_map"][value] for key, value in answers.items()}
    flags = [
        rule["item_id"]
        for rule in questionnaire.scoring_rules.get("safety_items", [])
        if scores.get(rule["item_id"], -1) >= rule["minimum_score"]
    ]
    total_ids = questionnaire.scoring_rules.get("total_item_ids", list(items))
    return sum(scores[k] for k in total_ids) if complete else None, flags


DEFAULT_SAFETY_GUIDANCE = (
    "В ответах отмечены мысли о причинении себе вреда. Пожалуйста, свяжитесь с врачом или "
    "специалистом по психическому здоровью сейчас для оценки безопасности. При непосредственной "
    "опасности обратитесь в экстренную службу."
)


def result_report(questionnaire, answers, auxiliary, score, flags, safety_guidance=None):
    """Собирает отчёт завершённой шкалы по сохранённым правилам без вызова моделей.

    При flags выбирает немедленную оценку безопасности независимо от суммы.
    Иначе учитывает порог следующего шага и диапазон интерпретации. Дополнительные
    ответы сохраняет отдельно, в сумму не включает; отчёт содержит версии и источники.
    """
    rules = questionnaire.scoring_rules
    band = next(
        (r for r in rules.get("thresholds", []) if r["minimum"] <= score <= r["maximum"]), None
    )
    followup_score = rules.get("followup_score")
    followup = followup_score is not None and score >= followup_score
    subscales = {
        code: sum(
            next(i for i in questionnaire.definition["items"] if i["id"] == key)["score_map"][
                answers[key]
            ]
            for key in ids
        )
        for code, ids in rules.get("subscales", {}).items()
    }
    review = {}
    if questionnaire.code == "PHQ9" and auxiliary.get("difficulty") in (2, 3):
        review["functional_impact"] = True
        followup = True  # Same functional-impact rule as chat; auxiliary stays outside the sum.
    if questionnaire.code == "CITYBITS":
        values = {
            i["id"]: i["score_map"][answers[i["id"]]] for i in questionnaire.definition["items"]
        }
        review = {
            "reexperiencing": any(values[str(n)] >= 1 for n in range(3, 8)),
            "avoidance": any(values[str(n)] >= 1 for n in range(8, 10)),
            "negative_mood": sum(values[str(n)] >= 1 for n in range(10, 17)) >= 2,
            "hyperarousal": sum(values[str(n)] >= 1 for n in range(17, 23)) >= 2,
            "distress_or_impairment": values["27"] >= 1 or values["28"] >= 1,
            "duration_at_least_month": values["26"] in (1, 2),
            "possible_other_cause": values["29"] >= 1,
        }
        # Повод для обсуждения с врачом, а не автоматическое заключение о ПТСР.
        followup = review["distress_or_impairment"] or all(
            review[k] for k in ("reexperiencing", "avoidance", "negative_mood", "hyperarousal")
        )
    if flags:
        action = "immediate_safety_assessment"
        next_step = safety_guidance or DEFAULT_SAFETY_GUIDANCE
    elif followup:
        action = "professional_assessment"
        next_step = (
            "Рекомендуется обсудить результат с врачом или специалистом по психическому здоровью."
        )
    else:
        action = "discuss_if_concerned"
        next_step = (
            "Если состояние беспокоит или мешает повседневной жизни, можно обратиться "
            "к специалисту при любом результате. Низкий балл не исключает трудностей."
        )
    return {
        "total_score": score,
        "maximum_score": rules["maximum"],
        "item_scores": {
            i["id"]: i["score_map"][answers[i["id"]]] for i in questionnaire.definition["items"]
        },
        "auxiliary_answers": auxiliary,
        "auxiliary_in_total": False,
        "subscales": subscales,
        "review_indicators": review,
        "interpretation": copy.deepcopy(band),
        "recommended_action": action,
        "next_step": next_step,
        "safety_flags": flags,
        "scoring_method": "sum_of_item_score_maps",
        "calculated_by": "python",
        "models_used": [],
        "scoring_version": rules.get("version", questionnaire.version),
        "source_references": rules.get("source_references", []),
        "interpretation_note": rules.get(
            "interpretation_note", "Самооценка симптомов, не диагноз."
        ),
        "not_a_diagnosis": True,
    }


class ScreeningEngine:
    def __init__(self, settings, jev):
        """Сохраняет настройки и Jev, загружает правила планирования шкал."""
        self.settings, self.jev = settings, jev
        self.rules = policy(settings, "screening_policy.yaml")

    async def owned(self, session, user_id, attempt_id):
        """Находит прохождение пользователя по UUID; при отсутствии возвращает HTTP 404."""
        attempt = await session.scalar(
            select(TestAttempt).where(TestAttempt.id == attempt_id, TestAttempt.user_id == user_id)
        )
        if attempt is None:
            raise HTTPException(404, "Прохождение не найдено")
        return attempt

    def safety_guidance(self, country=None, code=None):
        """Получает шаблон помощи при safety-ответе с контактами заданной страны."""
        kind = "citybits_safety" if code == "CITYBITS" else "questionnaire_safety"
        guard = SafetyGuard(self.settings, self.jev)
        if kind == "citybits_safety" and "citybits_safety_template" not in guard.rules:
            kind = "professional"
        return guard.template(kind, country)

    def view(self, attempt, questionnaire, country=None):
        """Собирает прохождение для API с исходной редакцией, прогрессом и результатом.

        Служебный хеш запроса скрывает. При safety-флагах добавляет помощь для текущей
        страны; сохранённый итоговый отчёт при этом не пересчитывает.
        """
        snapshot = attempt.questionnaire_snapshot or questionnaire_view(questionnaire)
        required = [i["id"] for i in snapshot["definition"]["items"]]
        data = public(attempt)
        data.pop("questionnaire_snapshot")
        context = {k: v for k, v in attempt.context.items() if k != "create_request_hash"}
        return {
            **data,
            "context": context,
            "questionnaire": snapshot,
            "questionnaire_code": snapshot["code"],
            "questionnaire_version": snapshot["version"],
            "questionnaire_hash": snapshot["definition_hash"],
            "progress": {
                "answered": len(set(attempt.answers) & set(required)),
                "required": len(required),
                "missing_item_ids": [i for i in required if i not in attempt.answers],
            },
            "safety_guidance": self.safety_guidance(country, snapshot["code"])
            if attempt.safety_flags
            else None,
            "next_step": attempt.result.get("next_step") if attempt.result else None,
            "not_a_diagnosis": True,
        }

    async def detail(self, session, user_id, attempt_id):
        """Загружает принадлежащее пользователю прохождение и возвращает его представление для API."""
        attempt = await self.owned(session, user_id, attempt_id)
        questionnaire = await session.get(Questionnaire, attempt.questionnaire_id)
        return self.view(attempt, questionnaire, await self.country(session, user_id))

    async def country(self, session, user_id):
        """Читает код страны из настроек пользователя для выбора контактов помощи."""
        return await session.scalar(
            select(UserSettings.country_code).where(UserSettings.user_id == user_id)
        )

    async def save(self, session, user_id, body, *, idempotent_completion=False):
        """Создаёт прохождение либо объединяет переданные ответы с сохранённым черновиком.

        Начальный client_request_id защищает от повтора, expected_revision — от
        перезаписи устаревшим клиентом. Считает по исходному снимку редакции; новые
        safety-флаги сохраняет сразу и не снимает при исправлении ответа. При complete
        проверяет полноту, записывает итог и обновляет расписание. Завершённая попытка
        неизменяема; idempotent_completion разрешает вернуть её при совпадении ответов.
        Commit и блокировка профиля остаются за вызывающей транзакцией.
        """
        is_new = body.attempt_id is None
        if (body.attempt_id and body.client_request_id) or (
            is_new and body.expected_revision is not None
        ):
            raise HTTPException(
                422, "client_request_id — для начала, expected_revision — для сохранения"
            )
        if body.attempt_id:
            attempt = await self.owned(session, user_id, body.attempt_id)
            questionnaire = await session.get(Questionnaire, attempt.questionnaire_id)
            if questionnaire.code not in SCALE_CODES:
                raise HTTPException(409, "Шкала снята с использования; доступна только история")
            if body.questionnaire_id and body.questionnaire_id != attempt.questionnaire_id:
                raise HTTPException(422, "Нельзя сменить версию опросника")
            if attempt.status == "completed":
                same_answers = all(attempt.answers.get(k) == v for k, v in body.answers.items())
                same_aux = all(
                    attempt.auxiliary_answers.get(k) == v for k, v in body.auxiliary_answers.items()
                )
                if idempotent_completion and same_answers and same_aux:
                    return self.view(attempt, questionnaire, await self.country(session, user_id))
                raise HTTPException(409, "Завершённое прохождение неизменяемо; начните новое")
            if body.expected_revision is not None and body.expected_revision != attempt.revision:
                raise HTTPException(
                    409,
                    {
                        "code": "revision_conflict",
                        "message": "Черновик изменён. Загрузите актуальную версию.",
                        "current_revision": attempt.revision,
                    },
                )
        else:
            if not body.questionnaire_id:
                raise HTTPException(422, "Укажите questionnaire_id или attempt_id")
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "questionnaire_id": str(body.questionnaire_id),
                        "answers": body.answers,
                        "auxiliary_answers": body.auxiliary_answers,
                        "complete": body.complete,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if body.client_request_id:
                existing = await session.scalar(
                    select(TestAttempt).where(
                        TestAttempt.user_id == user_id,
                        TestAttempt.client_request_id == body.client_request_id,
                    )
                )
                if existing:
                    if existing.context.get("create_request_hash") != fingerprint:
                        raise HTTPException(
                            409, "client_request_id уже использован с другим запросом"
                        )
                    return await self.detail(session, user_id, existing.id)
            questionnaire = await session.get(Questionnaire, body.questionnaire_id)
            if not questionnaire:
                raise HTTPException(404, "Опросник не найден")
            if not questionnaire.enabled:
                raise HTTPException(409, "Опросник отключён; новые прохождения недоступны")
            settings = await session.get(UserSettings, user_id)
            perinatal = perinatal_context(settings, utcnow())
            if questionnaire.code not in SCALE_CODES:
                raise HTTPException(409, "Шкала снята с использования; доступна только история")
            if questionnaire.code not in STAGE_SCALES[perinatal.stage]:
                raise HTTPException(409, "Шкала не применяется на текущем этапе")
            attempt = TestAttempt(
                user_id=user_id,
                questionnaire_id=questionnaire.id,
                pregnancy_id=settings.pregnancy_id,
                answers={},
                auxiliary_answers={},
                revision=0,
                client_request_id=body.client_request_id,
                safety_flags=[],
                questionnaire_snapshot=copy.deepcopy(questionnaire_view(questionnaire)),
                context={
                    # Срок на момент прохождения, а не когда-то переданное число.
                    "gestational_week": perinatal.gestational_week,
                    "perinatal_stage": perinatal.stage,
                    "postpartum_weeks": perinatal.postpartum_weeks,
                    "birth_date": settings.birth_date.isoformat() if settings.birth_date else None,
                    "create_request_hash": fingerprint,
                },
            )
            session.add(attempt)
        snapshot = attempt.questionnaire_snapshot or questionnaire_view(questionnaire)
        # A draft always uses its original version, even after a new registry version is published.
        material = SimpleNamespace(**snapshot)
        try:
            validate_material(snapshot)
        except (InvalidMaterial, KeyError, TypeError, ValueError):
            raise HTTPException(409, "Конфигурация опросника не прошла проверку") from None
        answers = {**attempt.answers, **body.answers}
        auxiliary = {**attempt.auxiliary_answers, **body.auxiliary_answers}
        aux_items = validated_answers(material.definition.get("auxiliary_items", []), auxiliary)
        if body.complete and any(
            i.get("required") and key not in auxiliary for key, i in aux_items.items()
        ):
            raise HTTPException(422, "Ответьте на обязательные дополнительные вопросы")
        score, flags = score_answers(material, answers, body.complete)
        new_flags = set(flags) - set(attempt.safety_flags)
        attempt.answers, attempt.auxiliary_answers = answers, auxiliary
        attempt.total_score = score
        attempt.safety_flags = sorted(set(attempt.safety_flags) | set(flags))
        if not is_new or body.answers or body.auxiliary_answers or body.complete:
            attempt.revision += 1
        await session.flush()
        if new_flags:
            attempt.context = {**attempt.context, "safety_observed_at": utcnow().isoformat()}
            SafetyGuard(self.settings, self.jev).record(
                session,
                user_id,
                None,
                "questionnaire_safety_item",
                "recommend_professional",
                "immediate_support_guidance",
            )
        country = await self.country(session, user_id)
        if body.complete:
            attempt.status, attempt.completed_at = "completed", utcnow()
            attempt.result = {
                **result_report(
                    material,
                    answers,
                    auxiliary,
                    score,
                    attempt.safety_flags,
                    self.safety_guidance(country, material.code),
                ),
                "questionnaire_hash": material_hash(snapshot),
                "completed_at": attempt.completed_at.isoformat(),
            }
            await self.completed_schedule(session, user_id, material.code, attempt)
        await session.flush()
        return self.view(attempt, questionnaire, country)

    async def completed_schedule(self, session, user_id, code, attempt):
        """Отмечает завершение шкалы и устанавливает срок допустимого повторного предложения.

        Интервал берёт из политики конкретной шкалы с учётом результата.
        Завершение снимает прежнюю отсрочку, история предложений сохраняется.
        """
        schedule = await session.scalar(
            select(AssessmentSchedule).where(
                AssessmentSchedule.user_id == user_id, AssessmentSchedule.questionnaire_code == code
            )
        )
        if schedule is None:
            schedule = AssessmentSchedule(user_id=user_id, questionnaire_code=code)
            session.add(schedule)
        now = utcnow()
        schedule.last_completed_at = now
        schedule.next_due_at = now + timedelta(
            days=ScreeningScheduler(self.settings).interval(code, attempt)
        )
        schedule.snoozed_until = None
        schedule.trigger_reason = "completed"
        schedule.policy_version = self.rules["version"]

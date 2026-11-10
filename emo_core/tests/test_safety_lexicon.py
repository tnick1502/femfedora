"""Deterministic first-line rules. Engineering expectations, not clinical labels."""

import json
from pathlib import Path

import pytest

from emo_core.providers.mock_providers import MockJevProvider
from emo_core.schemas.emotion import EmotionAnalysis
from emo_core.services.safety_guard import SafetyGuard
from emo_core.services.safety_lexicon import assess

DIALOGUES = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[1] / "docs/safety_dialogues.ru.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


@pytest.mark.parametrize(
    "row", [r for r in DIALOGUES if r["role"] == "user"], ids=lambda r: r["id"]
)
def test_user_dialogues_match_engineering_candidate(row):
    """Проверяет словарную маршрутизацию размеченных пользовательских реплик из набора сценариев."""
    route, expected = assess(row["text"]).route, row["engineering_candidate_route"]
    if expected == "review":
        # Решение оставлено классификатору и эксперту; правило не должно объявлять срочность.
        assert route != "urgent"
    else:
        assert route == expected


@pytest.mark.parametrize(
    "row", [r for r in DIALOGUES if r["role"] == "assistant"], ids=lambda r: r["id"]
)
def test_assistant_dialogues(row):
    """Проверяет программную блокировку ответов по размеченным сценариям набора."""
    blocked = not SafetyGuard.deterministic_output(row["text"])
    assert blocked == (row["engineering_candidate_route"] == "blocked_output")


@pytest.mark.parametrize(
    "text,risk",
    [
        ("Я хочу убить себя сейчас", "self_harm_intent"),
        ("Я уже приняла много таблеток", "self_harm_act"),
        ("Не вижу смысла жить", "self_harm_ideation"),
        ("Голоса приказывают мне навредить себе", "psychosis_command"),
        ("У меня сильное кровотечение", "acute_medical"),
        ("Какие таблетки от тревоги можно мне принимать?", "medication_request"),
    ],
)
def test_risk_codes(text, risk):
    """Проверяет код риска для заданного явного признака в тексте."""
    assert risk in assess(text).risks


def test_intrusive_thoughts_are_not_intent_but_acts_stay_urgent():
    """Проверяет различие навязчивых мыслей и уже совершённых опасных действий."""
    assert assess("Меня пугают навязчивые мысли, что я хочу навредить ребёнку").route == (
        "recommend_professional"
    )
    assert assess("У меня навязчивые мысли, и я уже порезала себя").route == "urgent"


def test_distant_past_is_left_to_classifier_but_recent_past_counts():
    """Проверяет исключение давних событий и срочный маршрут для недавних действий."""
    assert assess("Год назад я порезала себя").route == "normal"
    assert assess("Вчера я порезала себя").route == "urgent"


@pytest.mark.parametrize(
    "markers,route",
    [
        ({"self_harm_intent": 0.9}, "urgent"),
        ({"possible_self_harm": 0.95}, "recommend_professional"),
        ({"possible_self_harm": 0.4}, "recommend_professional"),
        # Откалибровано на живом Jev: слабый сигнал 0,16 был у «не хочется просыпаться».
        ({"possible_self_harm": 0.2}, "recommend_professional"),
        ({"possible_self_harm": 0.1}, "normal"),
        ({"request_for_medical_advice": 0.9}, "recommend_professional"),
        ({"request_for_medical_advice": 0.4}, "normal"),
        ({"possible_urgent_medical_issue": 0.9}, "urgent"),
        ({"possible_urgent_medical_issue": 0.4}, "recommend_professional"),
        ({"possible_violence_exposure": 0.9}, "recommend_professional"),
        ({"possible_violence_exposure": 0.4}, "normal"),
        ({"maternal_estrangement": 0.9}, "normal"),
    ],
)
def test_classifier_markers_route(config, markers, route):
    """Проверяет соответствие маркеров Jev порогам маршрутизации."""
    guard = SafetyGuard(config, MockJevProvider())
    decision = guard.assess_input("Нейтральный текст", EmotionAnalysis(markers=markers))
    assert decision.route == route


def test_rule_is_never_overridden_by_calm_classifier(config):
    """Проверяет, что низкие оценки Jev не отменяют явное срочное правило."""
    guard = SafetyGuard(config, MockJevProvider())
    decision = guard.assess_input(
        "Я хочу убить себя сейчас", EmotionAnalysis(markers={"self_harm_intent": 0.0})
    )
    assert decision.route == "urgent" and "deterministic_urgent_rule" in decision.reasons


def test_unavailable_classifier_is_not_treated_as_safe(config):
    """Проверяет рекомендацию специалиста при недоступном анализе обычного сообщения."""
    guard = SafetyGuard(config, MockJevProvider())
    decision = guard.assess_input("Привет", EmotionAnalysis(available=False))
    assert decision.route == "recommend_professional"


def test_templates_name_country_explicitly(config):
    """Проверяет указание страны контактов и отсутствие российских номеров в чужом fallback."""
    guard = SafetyGuard(config, MockJevProvider())
    assert "112" in guard.template("urgent", "RU") and "103" in guard.template("urgent", "RU")
    assert "В России" in guard.template("urgent", None)  # страна по умолчанию названа явно
    other = guard.template("urgent", "KZ")
    assert "103" not in other and "{emergency}" not in other
    assert "скорую" in guard.template("medical_urgent", "RU")


def test_weak_self_harm_signal_refers_without_crisis_text(config):
    """Проверяет различие слабого и уверенного сигнала для состава рисков и шаблона."""
    guard = SafetyGuard(config, MockJevProvider())
    weak = guard.assess_input(
        "Нейтральный текст", EmotionAnalysis(markers={"possible_self_harm": 0.2})
    )
    assert weak.risks == ["self_harm_uncertain"] and not weak.self_harm
    strong = guard.assess_input(
        "Нейтральный текст", EmotionAnalysis(markers={"possible_self_harm": 0.5})
    )
    assert strong.self_harm


def test_possible_psychosis_gets_same_day_assessment_template(config):
    """Проверяет выбор отдельного срочного шаблона при признаках изменения психического состояния."""
    guard = SafetyGuard(config, MockJevProvider())
    decision = guard.assess_input(
        "Нейтральный текст", EmotionAnalysis(markers={"possible_psychosis": 0.9})
    )
    assert decision.route == "urgent" and guard.urgent_kind(decision) == "mental_state_urgent"
    assert "врачу сегодня" in guard.template("mental_state_urgent", "RU")
    with_intent = guard.assess_input(
        "Нейтральный текст",
        EmotionAnalysis(markers={"possible_psychosis": 0.9, "self_harm_intent": 0.9}),
    )
    assert guard.urgent_kind(with_intent) == "urgent"


@pytest.mark.parametrize(
    ("text", "route"),
    [
        ("Меня пугают навязчивые картинки, будто я роняю малыша", "recommend_professional"),
        ("Постоянно представляю, как бросаю дочку с балкона, это ужасно", "recommend_professional"),
        ("Навязчивые мысли, что с ребёнком что-то случится", "recommend_professional"),
        ("Боюсь, что уроню ребёнка, когда несу по лестнице", "normal"),
        ("Коляску трясу, чтобы малыш уснул", "normal"),
        ("Трясла погремушкой перед малышом", "normal"),
        ("Била посуду при ребёнке", "normal"),
        ("Малыш ударился головой, плачет", "normal"),
        ("Я трясу малыша, когда он кричит", "urgent"),
        ("Сильно трясла его, малыш кричал", "urgent"),
        ("Я ударила своего ребёнка", "urgent"),
    ],
)
def test_harm_to_baby_wording(text, route):
    """Проверяет маршрутизацию разных формулировок риска причинения вреда ребёнку."""
    assert assess(text).route == route

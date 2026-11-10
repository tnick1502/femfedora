"""Граница приватности: наружу уходит обезличенный текст, пользователю возвращается исходный."""

import pytest
from sqlalchemy import select

from emo_core.db.models import Message
from emo_core.providers.mock_providers import MockJevProvider
from emo_core.services.anonymizer import PrivacyBoundary, boundary_for
from tests.conftest import conversation, send

P = "/api/v1/emo"
PUBLIC = ("88002000492", "112", "103")


def test_roundtrip_and_entity_types():
    """Проверяет маскирование типов персональных данных и точное восстановление исходного текста."""
    boundary = PrivacyBoundary(protected_terms=["Химки"], public_numbers=PUBLIC)
    text = (
        "Меня зовут Гульнара, мужа зовут Тимур. Живу в Химках, ул. Ленина, д. 5, кв. 12. "
        "Мой врач Соколова из роддома №7. Телефон мамы 8 (916) 123-45-67, почта nata.p@mail.ru. "
        "Полис 7700 1234 5678 9012, СНИЛС 123-456-789 01, родилась 12.03.1994. Пишите в @gulya_mama."
    )
    masked = boundary.mask(text)
    for secret in (
        "Гульнара",
        "Тимур",
        "Химках",
        "Ленина",
        "Соколова",
        "роддома №7",
        "916",
        "nata.p",
        "7700",
        "123-456",
        "12.03.1994",
        "gulya",
    ):
        assert secret not in masked, secret
    assert boundary.unmask(masked) == text
    assert boundary.report.as_dict()["entities"] == {
        "address": 1,
        "date": 1,
        "email": 1,
        "facility": 1,
        "link": 1,
        "name": 3,
        "number": 2,
        "phone": 1,
        "protected_term": 1,
    }
    assert "Гульнара" not in str(boundary.report.as_dict())


def test_clinical_content_and_public_numbers_are_kept():
    """Проверяет сохранение медицинского смысла и общедоступных экстренных номеров."""
    boundary = PrivacyBoundary(public_numbers=PUBLIC)
    text = (
        "Срок 30 недель, давление 120 на 80, EPDS 12 баллов, ПДР 15.01.2027. "
        "Вера в себя пропала. Позвонила на 8 800 2000 492 и на 112."
    )
    assert boundary.mask(text) == text


def test_same_person_keeps_one_number_across_forms_and_messages():
    """Проверяет общую нумерацию имени в разных формах и неизменность неизвестных меток."""
    boundary = PrivacyBoundary()
    first = boundary.mask("Саша опять кричал")
    second = boundary.mask("Я сказала Саше, что устала")
    assert "[ИМЯ_1]" in first and "[ИМЯ_1.2]" in second
    assert boundary.unmask("Поговори с [ИМЯ_1.2] и с [ИМЯ_1]") == "Поговори с Саше и с Саша"
    # Неизвестное обозначение не подменяется выдуманным значением.
    assert boundary.unmask("[ИМЯ_9]") == "[ИМЯ_9]"


def test_ambiguous_names_and_levels():
    """Проверяет контекст неоднозначных имён, строгий уровень и отключённое маскирование."""
    standard = PrivacyBoundary()
    assert "[ИМЯ_1]" in standard.mask("А Вера, моя сестра, говорит держаться")
    strict = PrivacyBoundary(level="strict")
    masked = strict.mask("Мы были в Сочи 12.05.2026")
    assert "Сочи" not in masked and "12.05.2026" not in masked
    assert PrivacyBoundary(enabled=False).mask("Саша, 8 916 123-45-67") == "Саша, 8 916 123-45-67"


def test_operator_policy_overrides_user_choice(config):
    """Проверяет приоритет серверных always/never над личным выбором пользователя."""
    config.anonymize_outbound = "always"
    assert boundary_for(config, {"privacy": {"anonymize": False}}).enabled
    config.anonymize_outbound = "never"
    assert not boundary_for(config, {"privacy": {"anonymize": True}}).enabled
    config.anonymize_outbound = "user"
    assert boundary_for(config, None).enabled  # по умолчанию включено


class SpyJev(MockJevProvider):
    def __init__(self):
        """Создаёт списки текстов, переданных на анализ и проверку ответа."""
        self.inputs, self.outputs = [], []

    async def analyze(self, text):
        """Запоминает исходящий текст и возвращает результат mock-анализа."""
        self.inputs.append(text)
        return await super().analyze(text)

    async def check_output(self, text, route):
        """Запоминает текст проверки ответа и вызывает mock-проверку."""
        self.outputs.append(text)
        return await super().check_output(text, route)


class SpyLLM:
    def __init__(self, reply):
        """Сохраняет заданный ответ и создаёт список запросов к тестовой модели."""
        self.reply, self.prompts = reply, []

    async def generate(self, messages, repair=False):
        """Запоминает контекст и возвращает заданный ответ с метками либо исходными именами."""
        self.prompts.append(messages)
        return self.reply


MESSAGE = (
    "Меня зовут Гульнара, муж Тимур совсем не помогает. Мой телефон 8 916 123-45-67, мне тревожно"
)


async def test_nothing_personal_leaves_and_reply_is_restored(client, auth, app, sessions):
    """Проверяет маскирование перед Jev/LLM и восстановление имён в сохранённом ответе."""
    app.state.jev, app.state.llm = (
        SpyJev(),
        SpyLLM("[ИМЯ_1], я слышу, как тебе тревожно. Что сейчас помогло бы?"),
    )
    cid = await conversation(client, auth)
    first = await send(client, auth, cid, MESSAGE)
    await send(
        client, auth, cid, "И ещё Тимур вчера накричал"
    )  # память и история тоже уходят наружу
    outbound = " ".join(
        [*app.state.jev.inputs, *app.state.jev.outputs]
        + [m["content"] for prompt in app.state.llm.prompts for m in prompt]
    )
    for secret in ("Гульнара", "Тимур", "916", "123-45-67"):
        assert secret not in outbound, secret
    assert "[ИМЯ_1]" in outbound and "тревожно" in outbound
    answer = first.json()["assistant_message"]
    assert answer["content"].startswith("Гульнара, я слышу")
    report = answer["metadata"]["decision_report"]
    privacy = next(s for s in report["stages"] if s["stage"] == "privacy")
    assert privacy["status"] == "applied" and privacy["facts"]["entities"]["name"] == 2
    assert privacy["facts"]["restored_in_reply"] == 1
    assert "Гульнара" not in str(report) and "Тимур" not in str(report)
    async with sessions() as s:  # в собственной базе хранится исходный текст
        stored = await s.scalar(select(Message).where(Message.role == "user").limit(1))
        assert "Гульнара" in stored.content


async def test_user_can_switch_anonymization_off_and_add_terms(client, auth, app):
    """Проверяет личный выбор маскирования и скрытие добавленных пользователем терминов."""
    app.state.jev, app.state.llm = SpyJev(), SpyLLM("Я рядом. Что сейчас важнее всего?")
    response = await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={
            "privacy": {
                "anonymize": True,
                "level": "standard",
                "protected_terms": ["Химки", " химки "],
            }
        },
    )
    assert response.status_code == 200 and response.json()["privacy"]["protected_terms"] == [
        "Химки"
    ]
    preview = await client.post(
        P + "/privacy/preview",
        headers=auth["anna"],
        json={"text": "Я живу в Химках, меня зовут Гульнара"},
    )
    assert preview.json()["text"] == "Я живу в [СКРЫТО_1], меня зовут [ИМЯ_1]"
    assert (
        app.state.jev.inputs == [] and app.state.llm.prompts == []
    )  # предпросмотр ничего не отправляет
    await client.patch(
        P + "/settings", headers=auth["anna"], json={"privacy": {"anonymize": False}}
    )
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, MESSAGE)).json()["assistant_message"]
    assert "Гульнара" in app.state.jev.inputs[0]
    privacy = next(
        s for s in answer["metadata"]["decision_report"]["stages"] if s["stage"] == "privacy"
    )
    assert privacy["status"] == "disabled"


@pytest.mark.parametrize(
    "body",
    [
        {"privacy": {"level": "paranoid"}},
        {"privacy": {"protected_terms": ["я"]}},
        {"privacy": {"protected_terms": ["x" * 61]}},
        {"context": {"compression": "sometimes"}},
        {"context": {"input_target_tokens": 3_000_000}},
        {"context": {"keep_recent_messages": 1}},
    ],
)
async def test_settings_validation(client, auth, body):
    """Проверяет отказ от некорректных личных параметров маскирования и контекста."""
    assert (await client.patch(P + "/settings", headers=auth["anna"], json=body)).status_code == 422


async def test_settings_expose_defaults_and_server_limits(client, auth):
    """Проверяет выдачу эффективных настроек AI и серверных ограничений."""
    data = (await client.get(P + "/settings", headers=auth["anna"])).json()
    assert data["privacy"] == {"anonymize": True, "level": "standard", "protected_terms": []}
    assert data["context"] == {
        "compression": "auto",
        "input_target_tokens": 0,
        "keep_recent_messages": 12,
    }
    assert data["limits"]["context_window_tokens"] == 256_000
    assert data["limits"]["hard_input_budget_tokens"] == 256_000 - 16_000 - 600
    assert set(data["limits"]["compression_modes"]) == {"off", "auto", "compact"}
    assert "ai_preferences" not in data


async def test_names_learned_in_one_message_are_hidden_later_without_hints(client, auth, app):
    """Проверяет повторное скрытие ранее обнаруженного имени без подсказки контекста."""
    app.state.jev, app.state.llm = SpyJev(), SpyLLM("Я рядом. Что сейчас важнее всего?")
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мою дочку зовут Айгерим, мне тревожно")
    other = await conversation(client, auth)
    await send(
        client, auth, other, "Айгерим опять не спала всю ночь"
    )  # без «зовут» и не из словаря
    assert "Айгерим" not in app.state.jev.inputs[-1]
    settings = (await client.get(P + "/settings", headers=auth["anna"])).json()
    assert settings["learned_names"] == ["Айгерим"]
    assert (
        await client.delete(P + "/privacy/learned-names", headers=auth["anna"])
    ).status_code == 200
    assert (await client.get(P + "/settings", headers=auth["anna"])).json()["learned_names"] == []


async def test_calibration_probes_are_gated(client):
    # Без входа в консоль проба доступна только там, где вход по паролю не настроен (тесты).
    """Проверяет dev-маршруты калибровки Jev и программные проверки ответа."""
    response = await client.post(
        "/dev/jev/check-output", json={"text": "Это просто гормоны, не нужно к врачу"}
    )
    assert response.status_code == 200 and response.json()["deterministic_passed"] is False
    assert (await client.post("/dev/jev/analyze", json={"text": "Мне тревожно"})).json()["markers"][
        "anxiety"
    ] == 1

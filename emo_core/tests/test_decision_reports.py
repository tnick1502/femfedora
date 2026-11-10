from uuid import uuid4

from emo_core.core.exceptions import ProviderUnavailable
from tests.conftest import conversation, send

P = "/api/v1/emo"


def stages(report):
    """Возвращает этапы отчёта решения по их именам."""
    return {stage["stage"]: stage for stage in report["stages"]}


async def test_report_matches_executed_rules_and_persists(client, auth):
    """Проверяет соответствие сохранённого отчёта реальному анализу, правилам и выбранному ответу."""
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Мне тревожно")
    assert response.status_code == 200, response.text
    answer = response.json()["assistant_message"]
    report = answer["metadata"]["decision_report"]
    steps = stages(report)
    assert steps["jev_analysis"]["facts"]["markers"]["anxiety"] == 1
    assert steps["jev_analysis"]["facts"]["source"] == "mock"
    assert steps["protocol_selection"]["facts"]["requested"] == "anxiety_support"
    assert steps["protocol_selection"]["facts"]["applied"] == "safe_baseline"
    assert steps["protocol_selection"]["reasons"] == ["requested_protocol_not_approved_or_missing"]
    assert report["models"]["jev"]["model"] == "deterministic-demo-v1"
    assert report["result"]["llm_calls"] == 1
    assert steps["output_safety"]["facts"]["attempts"][0]["jev_available"] is True
    path = f"{P}/conversations/{cid}/messages/{answer['id']}/decision-report"
    assert (await client.get(path, headers=auth["anna"])).json() == report
    assert (await client.get(path, headers=auth["maria"])).status_code == 404
    assert (await client.get(path)).status_code == 401
    assert "Мне тревожно" not in str(report)
    assert report["total_ms"] >= 0


async def test_replayed_message_keeps_original_report(client, auth):
    """Проверяет возврат исходного отчёта при повторе UUID сообщения."""
    cid = await conversation(client, auth)
    mid = str(uuid4())
    first = (await send(client, auth, cid, "Привет", client_message_id=mid)).json()
    second = (await send(client, auth, cid, "Привет", client_message_id=mid)).json()
    assert second["replayed"]
    assert (
        first["assistant_message"]["metadata"]["decision_report"]
        == second["assistant_message"]["metadata"]["decision_report"]
    )


async def test_urgent_report_does_not_claim_model_calls(client, auth):
    """Проверяет нулевые счётчики моделей при выборе срочного шаблона правилами."""
    cid = await conversation(client, auth)
    report = (await send(client, auth, cid, "Я хочу убить себя сейчас")).json()[
        "assistant_message"
    ]["metadata"]["decision_report"]
    assert report["result"]["llm_calls"] == 0
    assert report["result"]["jev_input_called"] is False
    assert report["result"]["jev_output_calls"] == 0
    assert stages(report)["jev_analysis"]["status"] == "skipped"
    assert stages(report)["protocol_selection"]["facts"]["applied"] is None


async def test_output_block_report_has_each_attempt(client, auth, app):
    """Проверяет наличие обеих заблокированных попыток генерации в отчёте."""

    class Unsafe:
        async def generate(self, *args, **kwargs):
            """Возвращает заведомо запрещённый ответ для проверки обеих попыток."""
            return "Ваш ребёнок точно здоров, гарантирую"

    app.state.llm = Unsafe()
    cid = await conversation(client, auth)
    report = (await send(client, auth, cid, "Привет")).json()["assistant_message"]["metadata"][
        "decision_report"
    ]
    checks = stages(report)["output_safety"]["facts"]["attempts"]
    assert len(checks) == 2
    assert all(c["reasons"] == ["deterministic_output_rule_blocked"] for c in checks)
    assert report["result"]["fallback"] == "output_safety_fallback"
    assert report["result"]["llm_calls"] == 2


async def test_jev_failure_report_sanitizes_error(client, auth, app, caplog):
    """Проверяет фиксированный код сбоя Jev без исходного секрета в ответе и логах."""
    secret = "private-provider-response-secret-4982"

    class Broken:
        async def analyze(self, text):
            """Имитирует сбой классификатора с секретом в тексте исключения."""
            raise ProviderUnavailable(secret)

    app.state.jev = Broken()
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Приватная реплика 3948")
    report = response.json()["assistant_message"]["metadata"]["decision_report"]
    step = stages(report)["jev_analysis"]
    assert step["status"] == "unavailable" and step["facts"]["markers"] == {}
    assert step["facts"]["error_code"] == "classifier_unavailable"
    assert secret not in response.text and secret not in caplog.text
    assert "Приватная реплика 3948" not in caplog.text

"""Менеджер контекста: бюджет окна, лестница сжатия, резюме ранней части беседы."""

from sqlalchemy import select

from emo_core.db.models import Conversation
from emo_core.services.context_manager import (
    SUMMARY_HEADER,
    SUMMARY_SYSTEM_PROMPT,
    TRUNCATION_NOTE,
    ChatContextManager,
    ContextBudget,
    estimate_tokens,
)
from tests.conftest import conversation, send

P = "/api/v1/emo"
SYSTEM = "Правила. " * 100


def history(count, size=900):
    """Создаёт чередующуюся историю user/assistant с заданными числом и размером реплик."""
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Реплика {i}. " + "текст " * (size // 6),
            "index": i,
        }
        for i in range(count)
    ]


def test_budget_follows_window_reserve_and_output():
    """Проверяет расчёт бюджета, ограничение личной цели и запасной режим сжатия."""
    budget = ContextBudget(
        context_window_tokens=256_000, safety_margin_tokens=16_000, max_output_tokens=600
    )
    assert budget.hard_input_budget == budget.target_input_budget == 239_400
    assert ContextBudget(input_target_tokens=20_000).target_input_budget == 20_000
    assert ContextBudget(input_target_tokens=10**9).target_input_budget == 239_400
    assert ContextBudget(compression_mode="unknown").compression_mode == "auto"
    assert estimate_tokens("а" * 28) == 10


def test_under_budget_nothing_is_lost():
    """Проверяет передачу всей истории без сжатия при достаточном бюджете."""
    plan = ChatContextManager(ContextBudget()).fit(
        SYSTEM, {"summary": "факт", "empty": []}, history(40)
    )
    assert plan.report.verdict == "ok" and not plan.to_summarize
    assert plan.report.history_messages_sent == 40 and plan.report.dropped_messages == 0
    assert plan.messages[0] == {"role": "system", "content": SYSTEM}


def test_ladder_goes_from_harmless_to_lossy_and_keeps_the_current_message():
    """Проверяет порядок сжатия, сохранение текущей реплики и неизменность исходной истории."""
    messages = history(60)
    manager = ChatContextManager(ContextBudget(input_target_tokens=6_000, keep_recent_messages=12))
    plan = manager.fit(SYSTEM, {"summary": "\n".join(f"строка {i}" for i in range(9))}, messages)
    steps = [step["step"] for step in plan.report.applied_steps]
    assert steps == [s for s in ChatContextManager.LADDER if s in steps]  # порядок лестницы
    assert "summary" in steps and len(plan.to_summarize) == 48
    assert plan.messages[-1]["content"] == messages[-1]["content"]
    assert plan.report.verdict == "compressed" and plan.report.estimated_prompt_tokens <= 6_000
    assert len(messages) == 60  # исходные данные не изменены

    refit = manager.fit(SYSTEM, {}, messages[48:], summary="Резюме ранней части")
    assert not refit.to_summarize and SUMMARY_HEADER in refit.messages[2]["content"]


def test_truncation_is_explicit_and_last_resort():
    """Проверяет явную пометку удаления старых реплик при недоступном резюме."""
    manager = ChatContextManager(ContextBudget(input_target_tokens=1_500, keep_recent_messages=4))
    plan = manager.fit(SYSTEM, {}, history(30), summary_available=False)
    assert plan.report.dropped_messages > 0 and not plan.to_summarize
    note = TRUNCATION_NOTE.format(dropped=plan.report.dropped_messages)
    assert plan.messages[2]["content"] == note
    assert plan.messages[-1]["content"].startswith("Реплика 29.")


def test_modes_off_and_compact():
    """Проверяет отказ от сжатия в off и обязательное резюме ранней истории в compact."""
    off = ChatContextManager(
        ContextBudget(context_window_tokens=8_192, safety_margin_tokens=0, compression_mode="off")
    )
    plan = off.fit(SYSTEM, {}, history(60))
    assert plan.report.verdict == "over_budget" and not plan.report.applied_steps
    compact = ChatContextManager(ContextBudget(compression_mode="compact", keep_recent_messages=6))
    plan = compact.fit(SYSTEM, {}, history(20))
    assert len(plan.to_summarize) == 14  # резюме всегда, даже когда окно не заполнено


def stages(response):
    """Извлекает этапы и полный отчёт решения для проверки бюджета контекста."""
    report = response.json()["assistant_message"]["metadata"]["decision_report"]
    return {s["stage"]: s for s in report["stages"]}, report


class Scripted:
    """Отличает запрос резюме от обычного ответа по системному промпту."""

    def __init__(
        self, summary="Она рассказывала о тревоге перед родами; помогает разговор с сестрой."
    ):
        """Задаёт ответ резюме и создаёт списки записанных запросов к модели."""
        self.summary, self.summary_requests, self.chat_prompts = summary, [], []

    async def generate(self, messages, repair=False):
        """Различает запрос резюме и обычный ответ, записывает контекст и имитирует заданный результат."""
        if messages[0]["content"] == SUMMARY_SYSTEM_PROMPT:
            self.summary_requests.append(messages[1]["content"])
            if self.summary is None:
                from emo_core.core.exceptions import ProviderUnavailable

                raise ProviderUnavailable("llm_unavailable")
            return self.summary
        self.chat_prompts.append(messages)
        return "Спасибо, что рассказываешь. Что сейчас важнее всего?"


async def compact_settings(client, auth, target=4_000, keep=2, mode="auto"):
    """Устанавливает бюджет, число недавних реплик и режим сжатия через API настроек."""
    response = await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={
            "context": {
                "compression": mode,
                "input_target_tokens": target,
                "keep_recent_messages": keep,
            }
        },
    )
    assert response.status_code == 200, response.text


async def test_long_conversation_is_summarized_once_and_summary_is_reused(
    client, auth, app, sessions
):
    """Проверяет сохранение и повторное использование резюме без повторной обработки старых реплик."""
    app.state.llm = llm = Scripted()
    await compact_settings(client, auth)
    cid = await conversation(client, auth)
    for i in range(6):
        response = await send(
            client, auth, cid, f"Сообщение {i}. " + "Мне тревожно перед родами. " * 60
        )
        assert response.status_code == 200
    steps, report = stages(response)
    assert len(llm.summary_requests) >= 1 and report["result"]["summary_llm_calls"] in (0, 1)
    assert "Сообщение 0." in llm.summary_requests[0]
    assert any(SUMMARY_HEADER in m["content"] for m in llm.chat_prompts[-1])
    assert llm.chat_prompts[-1][-1]["content"].startswith("Сообщение 5.")
    facts = steps["context"]["facts"]
    assert facts["estimated_prompt_tokens"] <= 4_000 and facts["within_budget"]
    async with sessions() as s:
        row = await s.scalar(select(Conversation))
        assert row.context_summary == llm.summary and row.context_summary_messages >= 2
        assert row.context_summary_until is not None
    # Свёрнутые реплики больше не загружаются и не пересказываются заново.
    before = len(llm.summary_requests)
    steps, _ = stages(await send(client, auth, cid, "Короткое сообщение"))
    assert steps["context"]["facts"]["history_messages_total"] < 13
    assert len(llm.summary_requests) == before
    # История для пользователя остаётся полной.
    history_ = (
        await client.get(f"{P}/conversations/{cid}/messages?limit=100", headers=auth["anna"])
    ).json()
    assert len(history_["items"]) == 14


async def test_failed_summary_falls_back_to_marked_truncation(client, auth, app):
    """Проверяет переход к помеченному усечению истории при сбое создания резюме."""
    app.state.llm = llm = Scripted(summary=None)
    await compact_settings(client, auth)
    cid = await conversation(client, auth)
    for i in range(6):
        response = await send(
            client, auth, cid, f"Сообщение {i}. " + "Мне тревожно перед родами. " * 60
        )
    steps, report = stages(response)
    assert report["result"]["origin"] == "reactive"
    assert any(
        step["step"] == "summary_failed" for step in steps["context"]["facts"]["applied_steps"]
    )
    assert steps["context"]["facts"]["dropped_messages"] > 0
    assert any("Контроль контекста" in m["content"] for m in llm.chat_prompts[-1])


async def test_unsafe_summary_is_discarded(client, auth, app, sessions):
    """Проверяет, что небезопасное резюме не сохраняется в беседе."""
    app.state.llm = Scripted(summary="У неё депрессия, принимайте препарат 20 мг в день.")
    await compact_settings(client, auth)
    cid = await conversation(client, auth)
    for i in range(6):
        await send(client, auth, cid, f"Сообщение {i}. " + "Мне тревожно перед родами. " * 60)
    async with sessions() as s:
        assert (await s.scalar(select(Conversation))).context_summary is None


async def test_compression_off_refuses_to_send_over_the_window(client, auth, app, config):
    """Проверяет отказ от вызова LLM при превышении окна в режиме off."""
    config.context_window_tokens, config.context_safety_margin_tokens = 8_192, 0
    app.state.llm = llm = Scripted()
    await compact_settings(client, auth, target=0, keep=12, mode="off")
    cid = await conversation(client, auth)
    for i in range(4):
        response = await send(
            client, auth, cid, f"Сообщение {i}. " + "Очень длинный рассказ. " * 500
        )
    steps, report = stages(response)
    assert (
        report["result"]["fallback"] == "context_over_budget" and report["result"]["llm_calls"] == 0
    )
    assert steps["context"]["status"] == "over_budget"
    assert "слишком длинной" in response.json()["assistant_message"]["content"]
    assert len(llm.chat_prompts) < 4

from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from emo_core.db.base import utcnow
from emo_core.db.models import Message, PromptTrace
from emo_core.services.prompt_trace import PromptRecorder, read_snapshot
from emo_core.services.retention import prune
from tests.conftest import conversation, send

P = "/api/v1/emo"


async def test_operator_can_disable_capture_without_disabling_chat(client, auth, config, sessions):
    config.prompt_trace_enabled = False
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Мне тревожно")
    assert response.status_code == 200
    manifest = response.json()["assistant_message"]["metadata"]["decision_report"]["prompt_trace"]
    assert manifest == {"available": False, "reason": "disabled_by_operator"}
    async with sessions() as session:
        assert await session.scalar(select(PromptTrace)) is None


async def test_snapshot_expires_before_message_and_is_pruned(client, auth, sessions, config):
    config.prompt_trace_retention_days = 30
    config.retention_days = 7
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Мне тревожно")).json()["assistant_message"]
    async with sessions() as session, session.begin():
        row = await session.scalar(select(PromptTrace))
        row.created_at = utcnow() - timedelta(days=8)
    assert (
        await client.get(
            f"{P}/conversations/{cid}/messages/{answer['id']}/prompt", headers=auth["anna"]
        )
    ).status_code == 404
    await prune(sessions, 7, prompt_days=30)
    async with sessions() as session:
        assert await session.scalar(select(PromptTrace)) is None
        assert await session.get(Message, UUID(answer["id"])) is not None


async def test_usage_counts_both_generations_even_when_prompt_capture_is_off(
    client, auth, app, config
):
    from emo_core.providers.llm_provider import LLMText

    config.prompt_trace_enabled = False
    unsafe = LLMText("У вас депрессия, я ставлю диагноз.")
    unsafe.usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    safe = LLMText("Спасибо, что написала. Что хочется обсудить?")
    safe.usage = {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135}
    app.state.llm.generate = AsyncMock(side_effect=[unsafe, safe])
    answer = (await send(client, auth, await conversation(client, auth), "Мне тревожно")).json()[
        "assistant_message"
    ]
    context = next(
        s["facts"]
        for s in answer["metadata"]["decision_report"]["stages"]
        if s["stage"] == "context"
    )
    assert context["provider_prompt_tokens"] == 220
    assert context["provider_completion_tokens"] == 25
    assert context["provider_usage_complete"] is True


async def test_prompt_snapshot_is_exact_private_and_not_in_history(client, auth, app, sessions):
    cid, request_id = await conversation(client, auth), str(uuid4())
    original = app.state.llm.generate
    app.state.llm.generate = AsyncMock(side_effect=original)
    response = await send(
        client, auth, cid, "Мне тревожно сегодня, хочется поговорить.", client_message_id=request_id
    )
    assert response.status_code == 200
    answer = response.json()["assistant_message"]
    manifest = answer["metadata"]["decision_report"]["prompt_trace"]
    assert manifest["available"] and manifest["requests"] == 1
    path = f"{P}/conversations/{cid}/messages/{answer['id']}/prompt"
    snapshot = (await client.get(path, headers=auth["anna"])).json()
    assert snapshot["requests"][0]["messages"] == app.state.llm.generate.call_args.args[0]
    assert snapshot["requests"][0]["purpose"] == "answer"
    assert snapshot["selected_system_prompt"] == snapshot["requests"][0]["messages"][0]["content"]
    assert (await client.get(path, headers=auth["maria"])).status_code == 404
    assert (await client.get(path)).status_code == 401
    other = await conversation(client, auth)
    assert (await client.get(path.replace(cid, other), headers=auth["anna"])).status_code == 404
    assert "selected_system_prompt" not in str(
        (await client.get(f"{P}/conversations/{cid}/messages", headers=auth["anna"])).json()
    )
    replay = await send(
        client, auth, cid, "Мне тревожно сегодня, хочется поговорить.", client_message_id=request_id
    )
    assert replay.json()["replayed"]
    assert (await client.get(path, headers=auth["anna"])).json() == snapshot
    async with sessions() as session:
        trace = await session.get(PromptTrace, UUID(answer["id"]))
        assert len(trace.compressed) < manifest["raw_bytes"]


async def test_urgent_snapshot_does_not_claim_a_model_request(client, auth, app):
    app.state.llm.generate = AsyncMock(side_effect=AssertionError("must not call"))
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Я хочу убить себя.")).json()["assistant_message"]
    snapshot = (
        await client.get(
            f"{P}/conversations/{cid}/messages/{answer['id']}/prompt", headers=auth["anna"]
        )
    ).json()
    assert snapshot["requests"] == []
    assert answer["metadata"]["decision_report"]["prompt_trace"]["model_called"] is False
    app.state.llm.generate.assert_not_awaited()


async def test_repair_snapshot_includes_actual_system_repair_instruction(client, auth, app):
    app.state.llm.generate = AsyncMock(
        side_effect=[
            "У вас депрессия, я ставлю диагноз.",
            "Я рядом. Что помогло бы сейчас почувствовать поддержку?",
        ]
    )
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Мне тревожно.")).json()["assistant_message"]
    snapshot = (
        await client.get(
            f"{P}/conversations/{cid}/messages/{answer['id']}/prompt", headers=auth["anna"]
        )
    ).json()
    assert [r["purpose"] for r in snapshot["requests"]] == ["answer", "answer_repair"]
    repair = snapshot["requests"][1]["messages"]
    assert (
        repair[1]["role"] == "system" and "Предыдущий ответ нарушил правила" in repair[1]["content"]
    )
    assert len(repair) == len(snapshot["requests"][0]["messages"]) + 1


async def test_prompt_lifecycle_matches_message_retention_and_profile_deletion(
    client, auth, sessions
):
    cid = await conversation(client, auth)
    await send(client, auth, cid, "Мне тревожно.")
    async with sessions() as session, session.begin():
        for message in (await session.scalars(select(Message))).all():
            message.created_at = utcnow() - timedelta(days=2)
    await prune(sessions, 1)
    async with sessions() as session:
        assert await session.scalar(select(PromptTrace)) is None
    await send(client, auth, cid, "Мне сегодня грустно.")
    assert (await client.delete(P + "/data", headers=auth["anna"])).status_code == 200
    async with sessions() as session:
        assert await session.scalar(select(PromptTrace)) is None


def test_snapshot_size_limit_is_explicit_and_integrity_checked(monkeypatch):
    recorder = PromptRecorder("инструкция", "mock")
    manifest, packed = recorder.freeze()
    assert read_snapshot(packed, manifest["sha256"])["selected_system_prompt"] == "инструкция"
    with pytest.raises(ValueError):
        read_snapshot(packed, "wrong-hash")
    monkeypatch.setattr("emo_core.services.prompt_trace.RAW_LIMIT", 1)
    manifest, packed = recorder.freeze()
    assert not manifest["available"] and packed is None
    assert manifest["reason"] == "prompt_snapshot_size_limit"

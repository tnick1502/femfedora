import asyncio
import io
import json
import wave
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import select

from emo_core.audio.gate import emotion_gate
from emo_core.audio.gigaam import GigaAM, verify
from emo_core.audio.ingestion import VoiceService
from emo_core.audio.normalize import Recording, normalize
from emo_core.audio.providers import BehavioralSignals, Deepgram, aggregate_emotions
from emo_core.core.exceptions import ProviderUnavailable
from emo_core.db.base import utcnow
from emo_core.db.models import Message, ModelRuntime, VoiceInput
from tests.conftest import conversation, send

P = "/api/v1/emo"


def wav(seconds=3, rate=16000):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(bytes(int(seconds * rate) * 2))
    return out.getvalue()


def provider_results(*, speaker="SPEAKER_00", start=0, end=3, sad=0.8):
    return [
        {"task": "diarization", "finalLabel": speaker},
        {
            "task": "emotion",
            "startTime": str(start),
            "endTime": str(end),
            "prediction": [
                {"label": k, "posterior": str(v)}
                for k, v in {
                    "sad": sad,
                    "angry": 0.1,
                    "happy": 0.05,
                    "neutral": 0.85 - sad,
                }.items()
            ],
        },
    ]


@pytest.fixture
def live_voice(app, config, tmp_path):
    service = VoiceService(
        config.model_copy(
            update={
                "voice_enabled": True,
                "ai_mode": "live",
                "external_ai_allowed": True,
                "deepgram_api_key": "secret-test-deepgram",
                "behavioral_signals_api_key": "secret-test-behavioral",
                "behavioral_signals_client_id": "42",
                "gigaam_model_dir": str(tmp_path / "models"),
            }
        )
    )
    service.deepgram.transcribe = AsyncMock(return_value="Я хочу сейчас спокойно поговорить.")
    service.gigaam.transcribe = AsyncMock(return_value="Я хочу сейчас спокойно поговорить.")
    service.behavioral.analyze = AsyncMock(
        return_value=aggregate_emotions({"modelVersion": "7.7.0", "results": provider_results()}, 3)
    )
    app.state.voice = service
    return service


async def upload(
    client,
    auth,
    cid,
    *,
    request_id=None,
    user="anna",
    backend="deepgram",
    data=None,
    consent=True,
    media="audio/wav",
):
    return await client.post(
        f"{P}/conversations/{cid}/voice-messages",
        params={
            "client_message_id": request_id or str(uuid4()),
            "backend": backend,
            "external_audio_consent": str(consent).lower(),
        },
        headers={**auth[user], "Content-Type": media},
        content=wav() if data is None else data,
    )


async def consent(client, auth):
    response = await client.patch(
        P + "/settings",
        headers=auth["anna"],
        json={"external_ai_consent": True, "voice": {"external_audio_consent": True}},
    )
    assert response.status_code == 200


async def test_admission_rejects_before_reading_audio(client, auth, live_voice):
    await consent(client, auth)
    cid = await conversation(client, auth)
    original = live_voice.read
    live_voice.read = AsyncMock(
        side_effect=AssertionError("must not read an upload without capacity")
    )
    async with live_voice.admission(), live_voice.admission():
        assert (await upload(client, auth, cid)).status_code == 429
    live_voice.read.assert_not_awaited()
    live_voice.read = original
    assert (await upload(client, auth, cid)).status_code == 200


async def test_failed_receipt_has_a_bounded_retry_budget(client, auth, live_voice, sessions):
    await consent(client, auth)
    cid, key = await conversation(client, auth), str(uuid4())
    live_voice.deepgram.transcribe.side_effect = ProviderUnavailable("test failure")
    for _ in range(3):
        assert (await upload(client, auth, cid, request_id=key)).status_code == 503
    assert (await upload(client, auth, cid, request_id=key)).status_code == 429
    assert live_voice.deepgram.transcribe.await_count == 3
    async with sessions() as session, session.begin():
        row = await session.scalar(select(VoiceInput))
        row.started_at = utcnow() - timedelta(hours=2)
    live_voice.deepgram.transcribe.side_effect = None
    assert (await upload(client, auth, cid, request_id=key)).status_code == 200
    assert (await upload(client, auth, cid, request_id=key)).status_code == 200
    assert live_voice.deepgram.transcribe.await_count == 4


async def test_request_flag_cannot_override_saved_audio_revocation(client, auth, live_voice):
    await consent(client, auth)
    await client.patch(
        P + "/settings", headers=auth["anna"], json={"voice": {"external_audio_consent": False}}
    )
    assert (
        await upload(client, auth, await conversation(client, auth), consent=True)
    ).status_code == 403
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_revocation_during_asr_keeps_transcript_and_skips_emotions(client, auth, live_voice):
    await consent(client, auth)

    async def transcribe(recording):
        await client.patch(
            P + "/settings", headers=auth["anna"], json={"voice": {"external_audio_consent": False}}
        )
        return "Я хочу сейчас спокойно поговорить."

    live_voice.deepgram.transcribe.side_effect = transcribe
    response = await upload(client, auth, await conversation(client, auth))
    assert response.status_code == 200
    assert response.json()["user_message"]["content"] == "Я хочу сейчас спокойно поговорить."
    assert (
        response.json()["user_message"]["metadata"]["voice"]["emotion"]["status"]
        == "consent_missing"
    )
    live_voice.behavioral.analyze.assert_not_awaited()


async def test_cancelled_local_inference_does_not_queue_more_cpu_jobs(config, monkeypatch):
    import threading

    model = GigaAM(config)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def transcribe(recording):
        calls.append(True)
        entered.set()
        release.wait(5)
        return "результат"

    monkeypatch.setattr(model, "_transcribe", transcribe)
    first = asyncio.create_task(model.transcribe(Recording(bytes(96000))))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        for _ in range(5):
            with pytest.raises(ProviderUnavailable, match="gigaam_busy"):
                await model.transcribe(Recording(bytes(96000)))
        assert len(calls) == 1
    finally:
        release.set()
        await model.running
    assert await model.transcribe(Recording(bytes(96000))) == "результат"
    assert len(calls) == 2


async def test_voice_uses_text_pipeline_and_replays_without_provider_calls(
    client, auth, app, live_voice, sessions
):
    await consent(client, auth)
    operation = (await client.get("/openapi.json")).json()["paths"][
        P + "/conversations/{conversation_id}/voice-messages"
    ]["post"]
    assert operation["requestBody"]["content"]["audio/webm"]["schema"]["format"] == "binary"
    cid, request_id = await conversation(client, auth), str(uuid4())
    generate = app.state.llm.generate
    app.state.llm.generate = AsyncMock(side_effect=generate)
    first = await upload(client, auth, cid, request_id=request_id)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["user_message"]["content"] == "Я хочу сейчас спокойно поговорить."
    assert data["user_message"]["metadata"]["voice"]["emotion"]["emotion"]["sad"] == 0.8
    report = data["assistant_message"]["metadata"]["decision_report"]
    state = next(s["facts"] for s in report["stages"] if s["stage"] == "state_evaluation")
    assert state["voice_clarification"] == "sad"
    assert state["recommend_specialist"] is False
    assert state["primary_support_need"] == "general"
    prompt = app.state.llm.generate.call_args.args[0]
    assert "не симптомы" in prompt[0]["content"] and "бережный вопрос" in prompt[0]["content"]
    second = await upload(client, auth, cid, request_id=request_id)
    assert second.status_code == 200 and second.json()["replayed"]
    assert second.json()["user_message"]["id"] == data["user_message"]["id"]
    live_voice.deepgram.transcribe.assert_awaited_once()
    live_voice.behavioral.analyze.assert_awaited_once()
    async with sessions() as session:
        rows = list(
            (
                await session.scalars(select(Message).where(Message.conversation_id == UUID(cid)))
            ).all()
        )
        assert len(rows) == 2
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert state["short"]["voice"]["current"] is None
    assert state["short"]["voice"]["recent"][0]["emotion"]["sad"] == 0.8
    text = await send(client, auth, cid, "Спасибо.")
    facts = next(
        s["facts"]
        for s in text.json()["assistant_message"]["metadata"]["decision_report"]["stages"]
        if s["stage"] == "state_evaluation"
    )
    assert facts["voice_clarification"] is None  # no repeated questioning from an old recording


async def test_urgent_transcript_never_waits_for_behavioral(client, auth, live_voice):
    await consent(client, auth)
    live_voice.deepgram.transcribe.return_value = "Я хочу убить себя."
    response = await upload(client, auth, await conversation(client, auth))
    assert response.status_code == 200
    assert response.json()["assistant_message"]["origin"] == "safety_template"
    assert response.json()["user_message"]["metadata"]["route"] == "urgent"
    assert (
        response.json()["user_message"]["metadata"]["voice"]["emotion"]["status"]
        == "skipped_safety"
    )
    live_voice.behavioral.analyze.assert_not_awaited()


async def test_emotion_outage_preserves_transcript_and_normal_response(client, auth, live_voice):
    await consent(client, auth)
    live_voice.behavioral.analyze.side_effect = ProviderUnavailable("no raw body")
    response = await upload(client, auth, await conversation(client, auth))
    assert response.status_code == 200
    assert response.json()["user_message"]["metadata"]["voice"]["emotion"] == {
        "status": "unavailable"
    }
    assert response.json()["assistant_message"]["origin"] == "reactive"


@pytest.mark.parametrize("consent_in_settings,consent_in_upload", [(False, True), (True, False)])
async def test_external_audio_requires_both_consents(
    client, auth, live_voice, consent_in_settings, consent_in_upload
):
    if consent_in_settings:
        await consent(client, auth)
    response = await upload(
        client, auth, await conversation(client, auth), consent=consent_in_upload
    )
    assert response.status_code == 403
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_voice_owner_and_auth_checked_before_audio_work(client, auth, live_voice):
    cid = await conversation(client, auth)
    assert (await upload(client, auth, cid, user="maria")).status_code == 404
    assert (
        await client.post(
            f"{P}/conversations/{cid}/voice-messages",
            params={"client_message_id": str(uuid4())},
            content=wav(),
        )
    ).status_code == 401
    live_voice.deepgram.transcribe.assert_not_awaited()


@pytest.mark.parametrize(
    "media,data,status",
    [
        ("text/plain", b"hello", 415),
        ("audio/wav", b"not audio", 422),
        ("audio/wav", b"", 422),
        ("audio/wav", wav(0.1), 422),
        ("audio/wav", wav(91), 422),
    ],
)
async def test_bad_audio_is_rejected_without_asr(client, auth, live_voice, media, data, status):
    await consent(client, auth)
    response = await upload(client, auth, await conversation(client, auth), media=media, data=data)
    assert response.status_code == status
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_upload_size_limit_is_enforced_for_chunked_body(client, auth, live_voice):
    await consent(client, auth)
    live_voice.settings.voice_max_bytes = 1024
    cid = await conversation(client, auth)

    async def body():
        yield bytes(700)
        yield bytes(700)

    response = await client.post(
        f"{P}/conversations/{cid}/voice-messages",
        params={"client_message_id": str(uuid4()), "external_audio_consent": True},
        headers={**auth["anna"], "Content-Type": "audio/wav"},
        content=body(),
    )
    assert response.status_code == 413
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_asr_failure_is_not_mocked_and_same_recording_can_retry(
    client, auth, live_voice, sessions
):
    await consent(client, auth)
    cid, request_id = await conversation(client, auth), str(uuid4())
    live_voice.deepgram.transcribe.side_effect = ProviderUnavailable("secret-test-deepgram")
    response = await upload(client, auth, cid, request_id=request_id)
    assert response.status_code == 503 and "secret-test" not in response.text
    async with sessions() as session:
        assert (await session.scalar(select(VoiceInput))).status == "failed"
        assert not list((await session.scalars(select(Message))).all())
    live_voice.deepgram.transcribe.side_effect = None
    assert (await upload(client, auth, cid, request_id=request_id)).status_code == 200


@pytest.mark.parametrize("transcript", ["", " ", "x" * 12001])
async def test_empty_or_oversized_asr_does_not_create_chat(client, auth, live_voice, transcript):
    await consent(client, auth)
    live_voice.deepgram.transcribe.return_value = transcript
    assert (await upload(client, auth, await conversation(client, auth))).status_code == 422
    live_voice.behavioral.analyze.assert_not_awaited()


async def test_idempotency_key_cannot_change_audio_or_be_reused_as_text(client, auth, live_voice):
    await consent(client, auth)
    cid, request_id = await conversation(client, auth), str(uuid4())
    assert (await upload(client, auth, cid, request_id=request_id)).status_code == 200
    assert (await upload(client, auth, cid, request_id=request_id, data=wav(4))).status_code == 409
    assert (
        await send(
            client, auth, cid, "Я хочу сейчас спокойно поговорить.", client_message_id=request_id
        )
    ).status_code == 409
    live_voice.deepgram.transcribe.assert_awaited_once()


async def test_local_gigaam_without_external_services_needs_no_external_consent(
    client, auth, live_voice, monkeypatch
):
    live_voice.settings.ai_mode = "disabled"
    monkeypatch.setattr(live_voice.gigaam, "installed", lambda: True)
    response = await upload(
        client, auth, await conversation(client, auth), backend="gigaam", consent=False
    )
    assert response.status_code == 200
    assert response.json()["user_message"]["metadata"]["voice"]["backend"] == "gigaam"
    live_voice.deepgram.transcribe.assert_not_awaited()
    live_voice.behavioral.analyze.assert_not_awaited()


async def test_history_is_bounded_stale_and_episode_scoped(client, auth, live_voice, sessions):
    await consent(client, auth)
    cid = await conversation(client, auth)
    for _ in range(4):
        assert (await upload(client, auth, cid)).status_code == 200
    state = (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"]
    assert len(state["short"]["voice"]["recent"]) == 3
    async with sessions() as session, session.begin():
        for row in (await session.scalars(select(Message))).all():
            row.created_at = utcnow() - timedelta(hours=73)
    assert (
        "voice"
        not in (await client.get(P + "/state", headers=auth["anna"])).json()["patient_context"][
            "short"
        ]
    )


async def test_receipts_deleted_with_user_data(client, auth, live_voice, sessions):
    await consent(client, auth)
    assert (await upload(client, auth, await conversation(client, auth))).status_code == 200
    assert (await client.delete(P + "/data", headers=auth["anna"])).status_code == 200
    async with sessions() as session:
        assert await session.scalar(select(VoiceInput)) is None


async def test_concurrent_retry_does_not_repeat_asr(client, auth, live_voice):
    await consent(client, auth)
    cid, request_id = await conversation(client, auth), str(uuid4())
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(recording):
        entered.set()
        await release.wait()
        return "Я хочу сейчас спокойно поговорить."

    live_voice.deepgram.transcribe.side_effect = transcribe
    first = asyncio.create_task(upload(client, auth, cid, request_id=request_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second = await upload(client, auth, cid, request_id=request_id)
        assert second.status_code == 409
    finally:
        release.set()
    assert (await first).status_code == 200
    live_voice.deepgram.transcribe.assert_awaited_once()


async def test_expired_lease_does_not_allow_an_old_request_to_overwrite_result(
    client, auth, live_voice, sessions
):
    await consent(client, auth)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def transcribe(recording):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            return "Предыдущая расшифровка."
        return "Я хочу сейчас спокойно поговорить."

    live_voice.deepgram.transcribe.side_effect = transcribe
    cid, request_id = await conversation(client, auth), str(uuid4())
    first = asyncio.create_task(upload(client, auth, cid, request_id=request_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with sessions() as session, session.begin():
            row = await session.scalar(select(VoiceInput))
            row.started_at = utcnow() - timedelta(minutes=6)
        replacement = await upload(client, auth, cid, request_id=request_id)
        assert replacement.status_code == 200
    finally:
        release.set()
    assert (await first).status_code == 409
    async with sessions() as session:
        row = await session.scalar(select(VoiceInput))
        assert row.status == "ready" and row.transcript == "Я хочу сейчас спокойно поговорить."


async def test_request_cancellation_releases_receipt_and_slot(client, auth, live_voice, sessions):
    await consent(client, auth)
    entered = asyncio.Event()

    async def transcribe(recording):
        entered.set()
        await asyncio.Event().wait()

    live_voice.deepgram.transcribe.side_effect = transcribe
    cid, request_id = await conversation(client, auth), str(uuid4())
    task = asyncio.create_task(upload(client, auth, cid, request_id=request_id))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with sessions() as session:
        assert (await session.scalar(select(VoiceInput))).status == "failed"
    assert not live_voice.slots.locked()
    live_voice.deepgram.transcribe.side_effect = None
    assert (await upload(client, auth, cid, request_id=request_id)).status_code == 200


async def test_runtime_mock_or_disabled_prevents_external_audio(client, auth, live_voice, sessions):
    await consent(client, auth)
    async with sessions() as session, session.begin():
        session.add(ModelRuntime(id=1, ai_mode="disabled"))
    caps = (await client.get(P + "/voice/config", headers=auth["anna"])).json()
    assert not caps["backends"]["deepgram"] and not caps["behavioral_signals"]
    assert (await upload(client, auth, await conversation(client, auth))).status_code == 503
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_gigaam_can_skip_behavioral_without_user_audio_consent(
    client, auth, live_voice, monkeypatch
):
    monkeypatch.setattr(live_voice.gigaam, "installed", lambda: True)
    response = await upload(
        client, auth, await conversation(client, auth), backend="gigaam", consent=False
    )
    assert response.status_code == 200
    assert (
        response.json()["user_message"]["metadata"]["voice"]["emotion"]["status"]
        == "consent_missing"
    )
    live_voice.behavioral.analyze.assert_not_awaited()


async def test_gigaam_short_form_limit_is_exposed_and_enforced(
    client, auth, live_voice, monkeypatch
):
    monkeypatch.setattr(live_voice.gigaam, "installed", lambda: True)
    caps = (await client.get(P + "/voice/config", headers=auth["anna"])).json()
    assert caps["backend_max_seconds"] == {"deepgram": 90, "gigaam": 25}
    response = await upload(
        client,
        auth,
        await conversation(client, auth),
        backend="gigaam",
        consent=False,
        data=wav(26),
    )
    assert response.status_code == 422
    live_voice.gigaam.transcribe.assert_not_awaited()


async def test_normalizer_downsamples_without_retaining_audio():
    recording = await normalize(wav(3, rate=44100), 90)
    assert recording.rate == 16000 and recording.seconds == 3
    with wave.open(io.BytesIO(recording.wav())) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000


@pytest.mark.parametrize(
    "container,codec,extra",
    [
        ("webm", "libopus", []),
        ("ogg", "libopus", []),
        ("mp4", "aac", ["-movflags", "frag_keyframe+empty_moov"]),
    ],
)
async def test_browser_recording_formats_decode_to_the_same_contract(container, codec, extra):
    import imageio_ffmpeg

    process = await asyncio.create_subprocess_exec(
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-c:a",
        codec,
        *extra,
        "-f",
        container,
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    data, _ = await process.communicate(wav())
    assert process.returncode == 0
    recording = await normalize(data, 90)
    assert 2.9 < recording.seconds < 3.2 and recording.rate == 16000


def test_posterior_aggregation_uses_duration_and_discards_demographics():
    first = provider_results(end=3, sad=0.8)
    second = provider_results(start=3, end=9, sad=0.2)
    result = aggregate_emotions(
        {
            "modelVersion": "7.7.0",
            "results": first + second + [{"task": "gender", "finalLabel": "female"}],
        },
        9,
    )
    assert result["emotion"]["sad"] == 0.4 and result["speech_seconds"] == 9
    assert "female" not in json.dumps(result)


def test_multiple_speakers_short_missing_or_invalid_results_are_not_emotions():
    assert (
        aggregate_emotions(provider_results() + provider_results(speaker="SPEAKER_01"), 3)["status"]
        == "multiple_speakers"
    )
    assert aggregate_emotions(provider_results(end=1), 1)["status"] == "insufficient_speech"
    broken = provider_results()
    broken[1]["prediction"][0]["posterior"] = "nan"
    assert aggregate_emotions(broken, 3)["status"] == "insufficient_speech"
    assert aggregate_emotions(provider_results() + provider_results(), 3)["speech_seconds"] == 3
    with pytest.raises(ProviderUnavailable):
        aggregate_emotions({"results": "invalid"}, 3)


async def test_deepgram_contract(config):
    config.deepgram_api_key = "secret-key"

    def handler(request):
        assert request.url.host == "api.deepgram.com" and request.url.params["language"] == "ru"
        assert request.url.params["model"] == "nova-3"
        assert (
            request.url.params["mip_opt_out"] == "true"
            and request.url.params["redact_usage"] == "true"
        )
        assert request.headers["Authorization"] == "Token secret-key"
        assert request.content.startswith(b"RIFF")
        return httpx.Response(
            200, json={"results": {"channels": [{"alternatives": [{"transcript": "Привет"}]}]}}
        )

    assert (
        await Deepgram(config, httpx.MockTransport(handler)).transcribe(Recording(bytes(96000)))
        == "Привет"
    )


async def test_behavioral_contract_envelope_posterior_and_polling(config):
    config.behavioral_signals_api_key, config.behavioral_signals_client_id = "secret-key", "42"
    calls = []

    def handler(request):
        assert request.url.host == "api.behavioralsignals.com"
        assert request.headers["X-Auth-Token"] == "secret-key"
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            assert b'filename="recording.wav"' in request.content
            assert b"embeddings" in request.content and b"false" in request.content
            return httpx.Response(200, json={"pid": 12, "status": 0})
        if request.url.path.endswith("/results"):
            return httpx.Response(
                200, json={"modelVersion": "7.7.0", "results": provider_results()}
            )
        return httpx.Response(200, json={"status": 2})

    result = await BehavioralSignals(config, httpx.MockTransport(handler)).analyze(
        Recording(bytes(96000))
    )
    assert result["emotion"]["sad"] == 0.8
    assert calls == [
        ("POST", "/v5/clients/42/processes/audio"),
        ("GET", "/v5/clients/42/processes/12"),
        ("GET", "/v5/clients/42/processes/12/results"),
    ]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_provider_errors_never_echo_tokens(config, status):
    def handler(request):
        return httpx.Response(status, text="secret-key and private transcript")

    with pytest.raises(ProviderUnavailable) as error:
        await Deepgram(config, httpx.MockTransport(handler)).transcribe(Recording(bytes(96000)))
    assert "secret" not in str(error.value) and "private" not in str(error.value)


def test_mock_mode_does_not_call_live_voice_providers(config):
    config.voice_enabled = config.external_ai_allowed = True
    config.deepgram_api_key = config.behavioral_signals_api_key = "secret-key"
    config.behavioral_signals_client_id = "42"
    caps = VoiceService(config).capabilities()
    assert caps["backends"]["deepgram"] is False and caps["behavioral_signals"] is False


def test_gigaam_checksums_and_missing_model(config, tmp_path):
    config.gigaam_model_dir = str(tmp_path)
    assert not GigaAM(config).installed()
    (tmp_path / "model.int8.onnx").write_bytes(b"wrong model")
    with pytest.raises(ProviderUnavailable):
        verify(tmp_path)


@pytest.mark.parametrize(
    "text,seconds,reason",
    [
        ("Мне грустно", 4, "short_speech_or_transcript"),
        ("Я хочу сейчас спокойно поговорить", 2, "short_speech_or_transcript"),
        ("Привет добрый день как дела спасибо большое", 6, "greeting_acknowledgement_or_mic_check"),
        (
            "Как включить микрофон в приложении на телефоне",
            6,
            "routine_interface_or_factual_request",
        ),
    ],
)
def test_gate_rejects_short_filler_and_routine_speech(text, seconds, reason):
    gate = emotion_gate(text, seconds)
    assert gate["eligible"] is False and gate["reason"] == reason


def test_routine_prefix_does_not_hide_substantive_speech():
    gate = emotion_gate("Как включить микрофон, мне сегодня очень грустно и одиноко", 8)
    assert gate["eligible"] and gate["reason"] == "substantive_speech"


async def test_short_transcript_skips_behavioral_but_still_uses_chat(client, auth, live_voice):
    await consent(client, auth)
    live_voice.deepgram.transcribe.return_value = "Мне грустно"
    response = await upload(client, auth, await conversation(client, auth))
    assert response.status_code == 200
    data = response.json()
    assert data["user_message"]["content"] == "Мне грустно"
    voice = data["user_message"]["metadata"]["voice"]
    assert voice["emotion"]["status"] == "skipped" and voice["emotion_provider_called"] is False
    assert voice["emotion_gate"]["reason"] == "short_speech_or_transcript"
    assert voice["timings_ms"]["transcription_ms"] >= 0
    live_voice.behavioral.analyze.assert_not_awaited()
    stages = data["assistant_message"]["metadata"]["decision_report"]["stages"]
    assert (
        next(s["facts"] for s in stages if s["stage"] == "jev_analysis")["markers"]["low_mood"] == 1
    )


async def test_user_can_disable_voice_without_affecting_other_profile(client, auth, live_voice):
    response = await client.patch(
        P + "/settings", headers=auth["anna"], json={"voice": {"enabled": False}}
    )
    assert response.status_code == 200 and response.json()["voice"]["enabled"] is False
    assert not (await client.get(P + "/voice/config", headers=auth["anna"])).json()["enabled"]
    assert (await client.get(P + "/voice/config", headers=auth["maria"])).json()["enabled"]
    assert (await upload(client, auth, await conversation(client, auth))).status_code == 503
    live_voice.deepgram.transcribe.assert_not_awaited()


async def test_admin_voice_model_and_global_switch_are_persisted_and_versioned(
    client, auth, live_voice, app
):
    app.state.settings.admin_api_token = "test-voice-admin-key-at-least-32-characters"
    headers = {"X-Admin-Token": app.state.settings.admin_api_token}
    registry = (await client.get(P + "/admin/models", headers=headers)).json()
    revision = registry["runtime"]["revision"]
    response = await client.patch(
        P + "/admin/model-runtime",
        headers=headers,
        json={"expected_revision": revision, "voice_enabled": False, "voice_backend": "gigaam"},
    )
    assert response.status_code == 200
    assert response.json()["voice"] == {"enabled": False, "backend": "gigaam"}
    for user in ("anna", "maria"):
        caps = (await client.get(P + "/voice/config", headers=auth[user])).json()
        assert not caps["enabled"] and caps["backend"] == "gigaam"
    assert (
        await client.patch(
            P + "/admin/model-runtime",
            headers=headers,
            json={"expected_revision": revision, "voice_enabled": True},
        )
    ).status_code == 409
    assert (
        await client.patch(
            P + "/admin/model-runtime",
            headers=auth["anna"],
            json={"expected_revision": revision + 1, "voice_enabled": True},
        )
    ).status_code == 403

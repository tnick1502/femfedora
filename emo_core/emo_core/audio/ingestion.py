import asyncio
import hashlib
import logging
from datetime import timedelta
from time import perf_counter
from uuid import uuid4

import anyio
from fastapi import HTTPException
from sqlalchemy import func, select

from emo_core.audio.gate import emotion_gate
from emo_core.audio.gigaam import MAX_SECONDS as GIGAAM_MAX_SECONDS
from emo_core.audio.gigaam import GigaAM
from emo_core.audio.normalize import normalize
from emo_core.audio.providers import BehavioralSignals, Deepgram
from emo_core.core.exceptions import ProviderUnavailable
from emo_core.db.base import utcnow
from emo_core.db.models import Message, ModelRuntime, VoiceInput
from emo_core.db.repositories import ensure_user, owned_conversation
from emo_core.schemas.user import VoiceSettings
from emo_core.services.admission import capacity, chat_budget
from emo_core.services.safety_lexicon import assess

LEASE_SECONDS = 300
log = logging.getLogger(__name__)
AUDIO_TYPES = frozenset(
    (
        "audio/webm",
        "video/webm",
        "audio/mp4",
        "video/mp4",
        "audio/ogg",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/flac",
        "audio/aac",
    )
)


class VoiceService:
    def __init__(self, settings):
        self.settings = settings
        self.gigaam = GigaAM(settings)
        self.deepgram = Deepgram(settings)
        self.behavioral = BehavioralSignals(settings)
        self.slots = asyncio.Semaphore(settings.voice_concurrency)

    def admission(self):
        # Reserve before reading bytes: concurrent slow uploads also consume memory.
        return capacity(self.slots, "Обработка голоса занята. Повторите через несколько секунд.")

    def capabilities(self, mode=None, *, enabled=None, backend=None):
        s = self.settings
        enabled = s.voice_enabled if enabled is None else enabled
        external = s.external_ai_allowed and (mode or s.ai_mode) == "live"
        return {
            "enabled": enabled,
            "backend": backend or s.voice_backend,
            "backends": {
                "deepgram": bool(enabled and external and s.deepgram_api_key),
                "gigaam": bool(enabled and self.gigaam.installed()),
            },
            "behavioral_signals": bool(
                enabled
                and external
                and s.behavioral_signals_api_key
                and s.behavioral_signals_client_id.isdecimal()
            ),
            "max_seconds": s.voice_max_seconds,
            "backend_max_seconds": {
                "deepgram": s.voice_max_seconds,
                "gigaam": min(s.voice_max_seconds, GIGAAM_MAX_SECONDS),
            },
            "max_bytes": s.voice_max_bytes,
            "audio_retained": False,
        }

    async def configuration(self, session, user=None):
        control = await session.get(ModelRuntime, 1)
        caps = self.capabilities(
            control.ai_mode if control else None,
            enabled=control.voice_enabled if control else None,
            backend=control.voice_backend if control else None,
        )
        caps["server_enabled"] = caps["enabled"]
        if user:
            preferences = VoiceSettings(**((user.ai_preferences or {}).get("voice") or {}))
            caps.update(
                user_enabled=preferences.enabled,
                external_audio_consent=preferences.external_audio_consent,
                user_ai_consent=user.external_ai_consent,
            )
            caps["enabled"] = caps["enabled"] and preferences.enabled
        return caps

    async def access(self, sessions, user_id, cid):
        async with sessions() as session, session.begin():
            user = await ensure_user(session, user_id, lock=False)
            await owned_conversation(session, user_id, cid)
            return await self.configuration(session, user)

    async def authorize(self, sessions, user_id, cid, backend, consent):
        capabilities = await self.access(sessions, user_id, cid)
        return self._allowed(backend, consent, capabilities)

    def _allowed(self, backend, consent, capabilities):
        if not capabilities["enabled"]:
            raise HTTPException(503, "Голосовые сообщения отключены")
        if not capabilities["backends"][backend]:
            raise HTTPException(503, "Выбранный способ распознавания пока не настроен")
        external = backend == "deepgram"
        if external and not (
            consent and capabilities["user_ai_consent"] and capabilities["external_audio_consent"]
        ):
            raise HTTPException(
                403,
                "Для отправки записи внешним сервисам нужно согласие на обработку AI в настройках и согласие на передачу аудио",
            )
        return capabilities

    async def read(self, request):
        s = self.settings
        if request.headers.get("content-type", "").split(";")[0].lower() not in AUDIO_TYPES:
            raise HTTPException(
                415, "Поддерживаются аудиозаписи WebM, MP4, Ogg, WAV, MP3, FLAC, AAC"
            )
        length = request.headers.get("content-length")
        if length and (not length.isdecimal() or int(length) > s.voice_max_bytes):
            raise HTTPException(413, "Запись слишком большая")
        data = bytearray()
        try:
            async with asyncio.timeout(20):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > s.voice_max_bytes:
                        raise HTTPException(413, "Запись слишком большая")
                    data.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, "Передача записи заняла слишком много времени") from None
        if not data:
            raise HTTPException(422, "Запись пустая")
        return bytes(data)

    async def prepare(self, sessions, user_id, cid, request_id, backend, consent, data):
        digest, lease, now = hashlib.sha256(data).hexdigest(), uuid4(), utcnow()
        async with sessions() as session, session.begin():
            user = await ensure_user(session, user_id)
            await owned_conversation(session, user_id, cid)
            self._allowed(backend, consent, await self.configuration(session, user))
            row = await session.scalar(
                select(VoiceInput).where(
                    VoiceInput.user_id == user_id,
                    VoiceInput.client_message_id == request_id,
                )
            )
            if row:
                if (row.conversation_id, row.audio_sha256, row.backend) != (cid, digest, backend):
                    raise HTTPException(409, "client_message_id уже использован для другой записи")
                if row.status == "ready":
                    return row.transcript, dict(row.details)
                if row.status == "processing" and row.started_at > now - timedelta(
                    seconds=LEASE_SECONDS
                ):
                    raise HTTPException(
                        409,
                        {
                            "code": "voice_processing",
                            "message": "Запись ещё обрабатывается. Повторите отправку через несколько секунд.",
                        },
                    )
            else:
                if await session.scalar(
                    select(Message.id).where(
                        Message.user_id == user_id,
                        Message.client_message_id == request_id,
                    )
                ):
                    raise HTTPException(409, "client_message_id уже использован")
            busy = await session.scalar(
                select(VoiceInput.id)
                .where(
                    VoiceInput.user_id == user_id,
                    VoiceInput.status == "processing",
                    VoiceInput.started_at > now - timedelta(seconds=LEASE_SECONDS),
                )
                .limit(1)
            )
            if busy:
                raise HTTPException(409, "Предыдущая запись ещё обрабатывается")
            count = await session.scalar(
                select(func.count())
                .select_from(VoiceInput)
                .where(
                    VoiceInput.user_id == user_id,
                    VoiceInput.created_at > now - timedelta(hours=1),
                )
            )
            if count >= 30 and row is None:
                raise HTTPException(429, "Слишком много записей. Попробуйте позже.")
            if row and row.started_at > now - timedelta(hours=1) and row.attempts >= 3:
                raise HTTPException(
                    429, "Лимит повторов этой записи исчерпан. Повторите через час."
                )
            await chat_budget(session, user_id, self.settings, request_id)
            if row is None:
                row = VoiceInput(
                    user_id=user_id,
                    conversation_id=cid,
                    client_message_id=request_id,
                    audio_sha256=digest,
                    backend=backend,
                    lease=lease,
                )
                session.add(row)
            row.attempts = (
                (row.attempts or 0) + 1
                if row.started_at and row.started_at > now - timedelta(hours=1)
                else 1
            )
            row.status, row.lease, row.started_at = "processing", lease, now
            # Persist episode at observation time; a later profile edit cannot reassign the cue.
            episode = str(user.pregnancy_id) if user.pregnancy_id else None
            await session.flush()
            receipt_id = row.id
        try:
            limit = (
                min(self.settings.voice_max_seconds, GIGAAM_MAX_SECONDS)
                if backend == "gigaam"
                else self.settings.voice_max_seconds
            )
            tick = perf_counter()
            recording = await normalize(data, limit)
            timings = {"decode_ms": round((perf_counter() - tick) * 1000, 2)}
            tick = perf_counter()
            async with asyncio.timeout(self.settings.voice_timeout_seconds):
                transcriber = self.deepgram if backend == "deepgram" else self.gigaam
                text = await transcriber.transcribe(recording)
            timings["transcription_ms"] = round((perf_counter() - tick) * 1000, 2)
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(
                    422, "Речь не распознана. Попробуйте записать ещё раз или напишите текст."
                )
            if len(text) > 12000:
                raise HTTPException(
                    422, "Расшифровка слишком длинная. Запишите более короткое сообщение."
                )
            # An urgent transcript proceeds immediately, without waiting for vocal analysis.
            urgent = assess(text).route == "urgent"
            gate = emotion_gate(text, recording.seconds)
            if urgent:
                gate = {**gate, "eligible": False, "reason": "urgent_text_bypass"}
            current = {}
            if urgent:
                emotion = {"status": "skipped_safety"}
            elif not gate["eligible"]:
                emotion = {"status": "skipped", "reason": gate["reason"]}
            else:
                # Revocation after ASR suppresses this optional layer, not the transcript.
                current = await self.access(sessions, user_id, cid)
                emotion = {"status": "not_configured"}
                if not current["enabled"]:
                    emotion = {"status": "disabled"}
                elif current["behavioral_signals"] and not (
                    consent and current["user_ai_consent"] and current["external_audio_consent"]
                ):
                    emotion = {"status": "consent_missing"}
            called = bool(
                gate["eligible"]
                and current.get("enabled")
                and current.get("behavioral_signals")
                and consent
                and current.get("user_ai_consent")
                and current.get("external_audio_consent")
            )
            tick = perf_counter()
            if called:
                try:
                    async with asyncio.timeout(self.settings.voice_timeout_seconds):
                        emotion = await self.behavioral.analyze(recording)
                except (ProviderUnavailable, TimeoutError):
                    emotion = {"status": "unavailable"}
            timings["emotion_ms"] = round((perf_counter() - tick) * 1000, 2)
            details = {
                "backend": backend,
                "model": "nova-3" if backend == "deepgram" else "gigaam-v3-e2e-ctc-int8",
                "seconds": round(recording.seconds, 1),
                "emotion": emotion,
                "emotion_gate": gate,
                "emotion_provider_called": called,
                "timings_ms": timings,
                "at": now.isoformat(timespec="seconds"),
                "pregnancy_id": episode,
                "external_audio_consent": bool(consent),
            }
            async with sessions() as session, session.begin():
                user = await ensure_user(session, user_id)
                await owned_conversation(session, user_id, cid)
                row = await session.get(VoiceInput, receipt_id)
                if row is None or row.lease != lease:
                    raise HTTPException(409, "Обработка записи уже возобновлена другим запросом")
                row.transcript, row.details, row.status = text.strip(), details, "ready"
            return text.strip(), details
        except BaseException as error:
            # Starlette cancels the whole request scope: asyncio.shield alone is insufficient.
            with anyio.CancelScope(shield=True):
                await self._fail(sessions, user_id, receipt_id, lease)
            if isinstance(error, (ProviderUnavailable, TimeoutError)):
                raise HTTPException(
                    503,
                    "Не удалось распознать речь. Повторите отправку или напишите текст.",
                ) from None
            raise

    @staticmethod
    async def _fail(sessions, user_id, receipt_id, lease):
        try:
            async with sessions() as session, session.begin():
                await ensure_user(session, user_id)
                row = await session.get(VoiceInput, receipt_id)
                if row and row.lease == lease and row.status == "processing":
                    row.status = "failed"
        except Exception:
            log.error("voice_interrupt_cleanup_failed")

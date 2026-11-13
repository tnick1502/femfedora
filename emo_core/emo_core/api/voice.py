from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from emo_core.api.conversations import send_message
from emo_core.core.auth import current_user
from emo_core.db.repositories import ensure_user
from emo_core.schemas.chat import MessageCreate

router = APIRouter()


@router.get("/voice/config")
async def config(request: Request, user_id: UUID = Depends(current_user)):
    service = request.app.state.voice
    async with request.app.state.sessions() as session, session.begin():
        user = await ensure_user(session, user_id, lock=False)
        return await service.configuration(session, user)


@router.post(
    "/conversations/{conversation_id}/voice-messages",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                mime: {"schema": {"type": "string", "format": "binary"}}
                for mime in ("audio/webm", "audio/mp4", "audio/ogg", "audio/wav")
            },
        }
    },
)
async def voice_message(
    conversation_id: UUID,
    request: Request,
    client_message_id: UUID,
    backend: Literal["deepgram", "gigaam"] = Query("deepgram"),
    external_audio_consent: bool = Query(False),
    user_id: UUID = Depends(current_user),
):
    """Raw audio body, bounded before decoding. Scores can only come from server adapters."""
    service, sessions = request.app.state.voice, request.app.state.sessions
    await service.authorize(sessions, user_id, conversation_id, backend, external_audio_consent)
    async with service.admission():
        data = await service.read(request)
        text, details = await service.prepare(
            sessions,
            user_id,
            conversation_id,
            client_message_id,
            backend,
            external_audio_consent,
            data,
        )
    return await send_message(
        request,
        user_id,
        conversation_id,
        MessageCreate(content=text, client_message_id=client_message_id),
        voice_details=details,
    )

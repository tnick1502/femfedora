from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.base import utcnow
from emo_core.db.models import Conversation, Message
from emo_core.db.repositories import owned_conversation, public, tenant_read_session, tenant_session
from emo_core.providers.factory import providers
from emo_core.schemas.chat import ConversationCreate, MessageCreate
from emo_core.services.chat_engine import ChatEngine
from emo_core.services.model_registry import ModelRegistry

router = APIRouter()


@router.post("/conversations", status_code=201)
async def create(
    body: ConversationCreate,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Создаёт беседу для владельца JWT и возвращает её сохранённые поля."""
    conversation = Conversation(user_id=user_id, **body.model_dump())
    session.add(conversation)
    await session.flush()
    return public(conversation)


@router.get("/conversations")
async def conversations(
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает страницу бесед владельца JWT, сначала недавно обновлённые."""
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id)
        .offset(offset)
        .limit(limit)
    )
    return {"items": [public(r) for r in rows], "limit": limit, "offset": offset}


@router.get("/conversations/{conversation_id}/messages")
async def messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает страницу сообщений собственной беседы в обратном порядке.

    items отсортированы от новых к старым; поле order равно newest_first.
    Перед чтением проверяется владелец беседы; чужой идентификатор даёт 404.
    """
    await owned_conversation(session, user_id, conversation_id)
    rows = await session.scalars(
        select(Message)
        .where(Message.user_id == user_id, Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return {
        "items": [public(r) for r in rows],
        "limit": limit,
        "offset": offset,
        "order": "newest_first",
    }


@router.post("/conversations/{conversation_id}/messages")
async def send(
    conversation_id: UUID,
    body: MessageCreate,
    request: Request,
    user_id: UUID = Depends(current_user),
):
    """Обрабатывает реплику с согласованным снимком общей конфигурации моделей.

    Снимок читается в короткой транзакции; ChatEngine сохраняет реплики
    и отчёт в своих транзакциях, освобождая соединение перед вызовом модели.
    client_message_id уникален для пользователя; смена текста или беседы
    при повторе этого ключа возвращает HTTP 409.
    """
    sessions = request.app.state.sessions
    async with sessions() as session, session.begin():
        snapshot = await ModelRegistry(request.app.state.settings).resolve(session)
    llm, jev = (
        providers(snapshot.settings)
        if snapshot.overridden
        else (request.app.state.llm, request.app.state.jev)
    )
    return await ChatEngine(snapshot.settings, llm, jev, snapshot.models).send(
        sessions, user_id, conversation_id, body
    )


@router.post("/conversations/{conversation_id}/read")
async def read(
    conversation_id: UUID,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_session, scope="function"),
):
    """Отмечает временем UTC все непрочитанные ответы ассистента в собственной беседе."""
    await owned_conversation(session, user_id, conversation_id)
    await session.execute(
        update(Message)
        .where(
            Message.user_id == user_id,
            Message.conversation_id == conversation_id,
            Message.role == "assistant",
            Message.read_at.is_(None),
        )
        .values(read_at=utcnow())
    )
    return {"status": "read"}


@router.get("/conversations/{conversation_id}/messages/{message_id}/decision-report")
async def decision_report(
    conversation_id: UUID,
    message_id: UUID,
    user_id: UUID = Depends(current_user),
    session: AsyncSession = Depends(tenant_read_session, scope="function"),
):
    """Возвращает сохранённый отчёт обработки ответа ассистента владельцу беседы.

    Для чужой реплики, отсутствующей беседы и сообщения без отчёта возвращает
    HTTP 404. Исторические отчёты при чтении не вычисляются заново.
    """
    await owned_conversation(session, user_id, conversation_id)
    message = await session.scalar(
        select(Message).where(
            Message.user_id == user_id,
            Message.conversation_id == conversation_id,
            Message.id == message_id,
            Message.role == "assistant",
        )
    )
    if message is None or "decision_report" not in message.details:
        raise HTTPException(404, "Отчёт для этого сообщения отсутствует")
    return message.details["decision_report"]

from uuid import UUID

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from emo_core.core.auth import current_user
from emo_core.db.models import Conversation, UserSettings


async def ensure_user(session: AsyncSession, user_id: UUID, lock: bool = True) -> UserSettings:
    """Находит или создаёт настройки пользователя и проверяет деактивацию.

    При lock=True берёт блокировку строки FOR UPDATE до конца транзакции.
    При lock=False существующую строку читает без блокировки. Создание
    защищено от повторной вставки; деактивированный профиль даёт HTTP 410.
    """
    query = select(UserSettings).where(UserSettings.user_id == user_id)
    result = None if lock else await session.scalar(query)
    if result is None:
        dialect = session.bind.dialect.name
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        await session.execute(insert(UserSettings).values(user_id=user_id).on_conflict_do_nothing())
        result = await session.scalar(query.with_for_update() if lock else query)
    if result.deleted_at:
        raise HTTPException(410, "Данные удалены, профиль EMO деактивирован")
    return result


async def tenant_session(request: Request, user_id: UUID = Depends(current_user)):
    """Открывает транзакцию записи и блокирует настройки владельца из JWT.

    Передаёт сессию через yield; операции одного пользователя выполняются
    последовательно. Успех фиксирует изменения, исключение откатывает их.
    """
    async with request.app.state.sessions() as session, session.begin():
        await ensure_user(session, user_id)
        yield session


async def tenant_read_session(request: Request, user_id: UUID = Depends(current_user)):
    """Открывает транзакцию чтения без блокировки строки пользователя.

    Проверяет деактивацию и при первом обращении создаёт настройки профиля.
    Предназначена для обработчиков без других записей: чтение не должно
    ждать транзакцию изменения настроек или сохранения результата.
    """
    async with request.app.state.sessions() as session, session.begin():
        await ensure_user(session, user_id, lock=False)
        yield session


async def owned_conversation(session, user_id, conversation_id):
    """Возвращает беседу, принадлежащую user_id.

    Для чужой и отсутствующей беседы возвращает одинаковый HTTP 404.
    """
    obj = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
    )
    if obj is None:
        raise HTTPException(404, "Беседа не найдена")
    return obj


def public(obj):
    """Преобразует колонки ORM-объекта в словарь без связей SQLAlchemy.

    Колонку metadata читает из атрибута details. Это общий сериализатор:
    он не скрывает секретные колонки, поэтому для профилей моделей нужны
    специальные представления ModelRegistry.
    """
    return {
        column.key if column.name != "metadata" else "metadata": getattr(
            obj, column.key if column.name != "metadata" else "details"
        )
        for column in obj.__table__.columns
    }

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from emo_core.core.config import get_settings
from emo_core.db import models  # noqa: F401
from emo_core.db.base import Base


def configure(connection):
    """Настраивает Alembic по ORM-метаданным и выполняет миграции на переданном соединении."""
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def online():
    """Открывает async-соединение из DATABASE_URL, выполняет миграции и освобождает пул."""
    engine = create_async_engine(get_settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(configure)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=get_settings().database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())

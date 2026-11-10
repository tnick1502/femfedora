from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from emo_core.core.config import get_settings


def make_engine(url: str, pool_size: int = 10, max_overflow: int = 20):
    """Создаёт асинхронный движок SQLAlchemy без вывода SQL-параметров.

    Для PostgreSQL задаёт размер пула, дополнительный резерв соединений
    и ожидание свободного соединения до 30 секунд. Для остальных DSN
    оставляет настройки пула драйверу; соединения проверяются перед выдачей.
    """
    options = {"pool_pre_ping": True, "echo": False, "hide_parameters": True}
    if url.startswith("postgresql"):
        # Предел соединений одного процесса; во время вызовов моделей чат освобождает БД.
        options.update(pool_size=pool_size, max_overflow=max_overflow, pool_timeout=30)
    return create_async_engine(url, **options)


_settings = get_settings()
engine = make_engine(_settings.database_url, _settings.db_pool_size, _settings.db_max_overflow)
Session = async_sessionmaker(engine, expire_on_commit=False)

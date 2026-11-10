from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    """Возвращает текущее время UTC с установленным часовым поясом."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Приводит дату к UTC перед записью в БД; None оставляет без изменений.

        Дата без часового пояса вызывает ValueError, чтобы локальное время
        не было сохранено как UTC по неявному предположению.
        """
        if value is not None:
            if value.tzinfo is None:
                raise ValueError("Timezone-aware UTC timestamp required")
            return value.astimezone(UTC)
        return None

    def process_result_value(self, value, dialect):
        """Возвращает прочитанную дату в UTC, сохраняя None.

        Если драйвер вернул дату без часового пояса, назначает ей UTC;
        дату с часовым поясом переводит в UTC.
        """
        if value is not None:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return None


class Base(DeclarativeBase):
    pass

from typing import Protocol
from uuid import UUID

GENERIC_PUSH = "У вас новое сообщение в приложении"


class NotificationProvider(Protocol):
    async def send(self, user_id: UUID, text: str, idempotency_key: str) -> None:
        """Доставляет уведомление пользователю с ключом защиты от повторной отправки.

        text содержит общий текст без медицинских подробностей; idempotency_key
        должен сохранять одну доставку при повторе вызова провайдера.
        """
        ...


class InAppOnlyProvider:
    async def send(self, user_id, text, idempotency_key):
        # In-app delivery is already committed. Push is optional and not configured.
        """Завершает вызов без push: сообщение внутри приложения уже сохранено отдельно."""
        return None

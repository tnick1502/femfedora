from datetime import datetime
from uuid import UUID

from pydantic import field_validator

from emo_core.schemas.user import StrictModel


class FollowupCreate(StrictModel):
    conversation_id: UUID
    scheduled_for: datetime
    consent: bool
    client_request_id: UUID

    @field_validator("scheduled_for")
    @classmethod
    def aware(cls, value):
        """Требует часовой пояс у scheduled_for; возвращает исходную дату."""
        if value.tzinfo is None:
            raise ValueError("Укажите часовой пояс")
        return value

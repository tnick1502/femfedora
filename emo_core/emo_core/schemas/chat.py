from typing import Literal
from uuid import UUID

from pydantic import Field

from emo_core.schemas.user import StrictModel


class ConversationCreate(StrictModel):
    title: str | None = Field(None, max_length=160)
    pregnancy_id: UUID | None = None


class MessageCreate(StrictModel):
    content: str = Field(min_length=1, max_length=12000)
    client_message_id: UUID


class MemoryPatch(StrictModel):
    summary: str = Field(max_length=3000)
    preferences: dict[str, str] = Field(default_factory=dict, max_length=20)


class FeedbackCreate(StrictModel):
    message_id: UUID
    rating: Literal["better", "unchanged", "worse", "skip"]

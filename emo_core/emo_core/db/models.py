from datetime import date, datetime, time
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    Time,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from emo_core.db.base import Base, UTCDateTime, utcnow

J = JSON().with_variant(JSONB(), "postgresql")


class Identified:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)


class Tenant:
    user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("emo_user_settings.user_id", ondelete="CASCADE"), index=True
    )


class Created:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class UserSettings(Base):
    __tablename__ = "emo_user_settings"
    user_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    timezone: Mapped[str] = mapped_column(String, default="Europe/Moscow")
    proactive_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    push_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    external_ai_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    quiet_hours_start: Mapped[time] = mapped_column(Time, default=time(20))
    quiet_hours_end: Mapped[time] = mapped_column(Time, default=time(9))
    proactive_paused_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    # user — пауза задана пользователем; unanswered — автоматическая после сообщений без ответа.
    proactive_pause_reason: Mapped[str | None] = mapped_column(String(24))
    pregnancy_id: Mapped[UUID | None] = mapped_column(Uuid)
    gestational_week: Mapped[int | None] = mapped_column(Integer)
    gestational_week_recorded_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    estimated_due_date: Mapped[date | None] = mapped_column(Date)
    birth_date: Mapped[datetime | None] = mapped_column(UTCDateTime())
    # ongoing | live_birth | loss; передаётся основным приложением.
    pregnancy_outcome: Mapped[str | None] = mapped_column(String(16))
    reproductive_stage: Mapped[str | None] = mapped_column(String(24))
    country_code: Mapped[str | None] = mapped_column(String(2))
    # Настройки личного кабинета: обезличивание и управление контекстом.
    ai_preferences: Mapped[dict] = mapped_column(J, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class Conversation(Identified, Tenant, Created, Base):
    __tablename__ = "emo_conversations"
    __table_args__ = (UniqueConstraint("id", "user_id"),)
    pregnancy_id: Mapped[UUID | None] = mapped_column(Uuid)
    title: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="active")
    # Резюме ранней части беседы, которой уже нет в контексте, и граница свёрнутых реплик.
    context_summary: Mapped[str | None] = mapped_column(Text)
    context_summary_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    context_summary_messages: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class Message(Identified, Tenant, Created, Base):
    __tablename__ = "emo_messages"
    __table_args__ = (
        UniqueConstraint("user_id", "client_message_id"),
        UniqueConstraint("id", "user_id"),
        ForeignKeyConstraint(
            ["conversation_id", "user_id"],
            ["emo_conversations.id", "emo_conversations.user_id"],
            ondelete="CASCADE",
        ),
        Index("ix_messages_history", "user_id", "conversation_id", "created_at"),
    )
    conversation_id: Mapped[UUID] = mapped_column(Uuid)
    role: Mapped[str] = mapped_column(String)
    origin: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    client_message_id: Mapped[UUID | None] = mapped_column(Uuid)
    delivery_status: Mapped[str] = mapped_column(String, default="delivered")
    protocol_id: Mapped[str | None] = mapped_column(String)
    protocol_version: Mapped[str | None] = mapped_column(String)
    read_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)


class VoiceInput(Identified, Tenant, Created, Base):
    """A resumable ingestion receipt. Raw audio and provider responses are never stored."""

    __tablename__ = "emo_voice_inputs"
    __table_args__ = (
        UniqueConstraint("user_id", "client_message_id"),
        Index("ix_voice_user_created", "user_id", "created_at"),
        Index("ix_voice_user_processing", "user_id", "status", "started_at"),
        ForeignKeyConstraint(
            ["conversation_id", "user_id"],
            ["emo_conversations.id", "emo_conversations.user_id"],
            ondelete="CASCADE",
        ),
    )
    conversation_id: Mapped[UUID] = mapped_column(Uuid)
    client_message_id: Mapped[UUID] = mapped_column(Uuid)
    audio_sha256: Mapped[str] = mapped_column(String(64))
    backend: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="processing")
    lease: Mapped[UUID] = mapped_column(Uuid)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    transcript: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(J, default=dict)


class PromptTrace(Tenant, Created, Base):
    """Compressed private prompt snapshots; read on demand, never via chat history."""

    __tablename__ = "emo_prompt_traces"
    __table_args__ = (
        ForeignKeyConstraint(
            ["message_id", "user_id"],
            ["emo_messages.id", "emo_messages.user_id"],
            ondelete="CASCADE",
        ),
    )
    message_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    compressed: Mapped[bytes] = mapped_column(LargeBinary)
    sha256: Mapped[str] = mapped_column(String(64))


class UserState(Tenant, Base):
    __tablename__ = "emo_user_state"
    __mapper_args__ = {"primary_key": ["user_id"]}
    user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("emo_user_settings.user_id", ondelete="CASCADE"), primary_key=True
    )
    memory_cleared_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    summary: Mapped[str] = mapped_column(Text, default="")
    preferences: Mapped[dict] = mapped_column(J, default=dict)
    recurring_topics: Mapped[list] = mapped_column(J, default=list)
    facts: Mapped[list] = mapped_column(J, default=list)
    last_summarized_message_id: Mapped[UUID | None] = mapped_column(Uuid)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class MessageSignal(Identified, Tenant, Base):
    __tablename__ = "emo_message_signals"
    __table_args__ = (
        ForeignKeyConstraint(
            ["message_id", "user_id"],
            ["emo_messages.id", "emo_messages.user_id"],
            ondelete="CASCADE",
        ),
    )
    message_id: Mapped[UUID] = mapped_column(Uuid)
    signal_code: Mapped[str] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String)
    model_version: Mapped[str | None] = mapped_column(String)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)


class Complaint(Identified, Tenant, Base):
    __tablename__ = "emo_complaints"
    __table_args__ = (UniqueConstraint("user_id", "category"),)
    pregnancy_id: Mapped[UUID | None] = mapped_column(Uuid)
    category: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="new")
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_user_reported_intensity: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)


class Checkin(Identified, Tenant, Created, Base):
    __tablename__ = "emo_checkins"
    anxiety_score: Mapped[int] = mapped_column(Integer)
    mood_score: Mapped[int] = mapped_column(Integer)
    sleep_quality: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String, default="user_report")


class Questionnaire(Identified, Created, Base):
    __tablename__ = "emo_questionnaires"
    __table_args__ = (UniqueConstraint("code", "version", "language"),)
    code: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, default="ru")
    definition: Mapped[dict] = mapped_column(J)
    scoring_rules: Mapped[dict] = mapped_column(J)
    license_reference: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)


class TestAttempt(Identified, Tenant, Base):
    __tablename__ = "emo_test_attempts"
    __table_args__ = (
        UniqueConstraint("user_id", "client_request_id", name="uq_attempt_user_request"),
    )
    pregnancy_id: Mapped[UUID | None] = mapped_column(Uuid)
    questionnaire_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("emo_questionnaires.id"))
    answers: Mapped[dict] = mapped_column(J, default=dict)
    auxiliary_answers: Mapped[dict] = mapped_column(J, default=dict)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    client_request_id: Mapped[UUID | None] = mapped_column(Uuid)
    questionnaire_snapshot: Mapped[dict | None] = mapped_column(J)
    result: Mapped[dict | None] = mapped_column(J)
    total_score: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="in_progress")
    safety_flags: Mapped[list] = mapped_column(J, default=list)
    context: Mapped[dict] = mapped_column(J, default=dict)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), index=True)


class AssessmentSchedule(Identified, Tenant, Base):
    __tablename__ = "emo_assessment_schedule"
    __table_args__ = (UniqueConstraint("user_id", "questionnaire_code"),)
    questionnaire_code: Mapped[str] = mapped_column(String)
    last_offered_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    next_due_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    snoozed_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    trigger_reason: Mapped[str] = mapped_column(String, default="initial")
    policy_version: Mapped[str] = mapped_column(String, default="1.0")
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ProactiveJob(Identified, Tenant, Created, Base):
    __tablename__ = "emo_proactive_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "user_id"],
            ["emo_conversations.id", "emo_conversations.user_id"],
            ondelete="CASCADE",
        ),
        Index("ix_jobs_due", "status", "scheduled_for"),
    )
    conversation_id: Mapped[UUID] = mapped_column(Uuid)
    job_type: Mapped[str] = mapped_column(String)
    deduplication_key: Mapped[str] = mapped_column(String, unique=True)
    scheduled_for: Mapped[datetime] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String, default="pending")
    message_id: Mapped[UUID | None] = mapped_column(Uuid)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)


class SafetyEvent(Identified, Tenant, Created, Base):
    __tablename__ = "emo_safety_events"
    message_id: Mapped[UUID | None] = mapped_column(Uuid)
    risk_type: Mapped[str] = mapped_column(String)
    risk_route: Mapped[str] = mapped_column(String)
    action_taken: Mapped[str] = mapped_column(String)
    policy_version: Mapped[str] = mapped_column(String)
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)


class NotificationOutbox(Identified, Tenant, Created, Base):
    __tablename__ = "emo_notification_outbox"
    message_id: Mapped[UUID] = mapped_column(Uuid, unique=True)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String)


class SupportFeedback(Identified, Tenant, Created, Base):
    __tablename__ = "emo_support_feedback"
    message_id: Mapped[UUID] = mapped_column(Uuid)
    rating: Mapped[str] = mapped_column(String)


class ModelProfile(Identified, Created, Base):
    __tablename__ = "emo_model_profiles"
    kind: Mapped[str] = mapped_column(String(8))
    name: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(160))
    base_url: Mapped[str] = mapped_column(String(500))
    timeout_seconds: Mapped[float] = mapped_column(Float)
    credential_ref: Mapped[str | None] = mapped_column(String(100))
    encrypted_api_key: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ModelRuntime(Base):
    __tablename__ = "emo_model_runtime"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    ai_mode: Mapped[str | None] = mapped_column(String(16))
    voice_enabled: Mapped[bool | None] = mapped_column(Boolean)
    voice_backend: Mapped[str | None] = mapped_column(String(16))
    llm_profile_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("emo_model_profiles.id"))
    jev_profile_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("emo_model_profiles.id"))
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ModelConfigEvent(Identified, Created, Base):
    __tablename__ = "emo_model_config_events"
    revision: Mapped[int] = mapped_column(Integer, index=True)
    action: Mapped[str] = mapped_column(String(40))
    profile_id: Mapped[UUID | None] = mapped_column(Uuid)
    details: Mapped[dict] = mapped_column("metadata", J, default=dict)

"""Resumable, tenant-scoped voice ingestion; no raw audio storage."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "e46b37a910cd"
down_revision = "d92c45e17a60"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "emo_voice_inputs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("emo_user_settings.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("client_message_id", sa.Uuid(), nullable=False),
        sa.Column("audio_sha256", sa.String(64), nullable=False),
        sa.Column("backend", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("lease", sa.Uuid(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("transcript", sa.Text()),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("user_id", "client_message_id"),
        sa.ForeignKeyConstraint(
            ["conversation_id", "user_id"],
            ["emo_conversations.id", "emo_conversations.user_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_emo_voice_inputs_user_id", "emo_voice_inputs", ["user_id"])
    op.create_index("ix_emo_voice_inputs_created_at", "emo_voice_inputs", ["created_at"])


def downgrade():
    op.drop_table("emo_voice_inputs")

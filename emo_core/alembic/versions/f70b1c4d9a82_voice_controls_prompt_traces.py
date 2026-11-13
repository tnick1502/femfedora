"""Shared voice controls and private compressed prompt snapshots."""

import sqlalchemy as sa

from alembic import op

revision = "f70b1c4d9a82"
down_revision = "e46b37a910cd"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("emo_model_runtime", sa.Column("voice_enabled", sa.Boolean(), nullable=True))
    op.add_column("emo_model_runtime", sa.Column("voice_backend", sa.String(16), nullable=True))
    op.create_table(
        "emo_prompt_traces",
        sa.Column("message_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("emo_user_settings.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("compressed", sa.LargeBinary(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id", "user_id"],
            ["emo_messages.id", "emo_messages.user_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_emo_prompt_traces_user_id", "emo_prompt_traces", ["user_id"])
    op.create_index("ix_emo_prompt_traces_created_at", "emo_prompt_traces", ["created_at"])


def downgrade():
    op.drop_table("emo_prompt_traces")
    op.drop_column("emo_model_runtime", "voice_backend")
    op.drop_column("emo_model_runtime", "voice_enabled")

"""Bound failed voice retries and index tenant quota/lease lookups."""

import sqlalchemy as sa

from alembic import op

revision = "a184c6d2f903"
down_revision = "f70b1c4d9a82"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "emo_voice_inputs", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0")
    )
    op.create_index("ix_voice_user_created", "emo_voice_inputs", ["user_id", "created_at"])
    op.create_index(
        "ix_voice_user_processing", "emo_voice_inputs", ["user_id", "status", "started_at"]
    )


def downgrade():
    op.drop_index("ix_voice_user_processing", table_name="emo_voice_inputs")
    op.drop_index("ix_voice_user_created", table_name="emo_voice_inputs")
    op.drop_column("emo_voice_inputs", "attempts")

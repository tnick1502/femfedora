"""privacy and context settings"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

import emo_core.db.base
from alembic import op

revision = "a7c2e94d1f36"
down_revision = "f3a91c5d2b74"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade():
    """Добавляет личные настройки AI и поля сохранённого резюме беседы без изменения сообщений."""
    op.add_column(
        "emo_user_settings",
        sa.Column("ai_preferences", JSON, nullable=False, server_default=sa.text("'{}'")),
    )
    op.alter_column("emo_user_settings", "ai_preferences", server_default=None)
    op.add_column("emo_conversations", sa.Column("context_summary", sa.Text(), nullable=True))
    op.add_column(
        "emo_conversations",
        sa.Column(
            "context_summary_until", emo_core.db.base.UTCDateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "emo_conversations",
        sa.Column("context_summary_messages", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("emo_conversations", "context_summary_messages", server_default=None)


def downgrade():
    """Удаляет настройки AI и сохранённые резюме бесед."""
    op.drop_column("emo_conversations", "context_summary_messages")
    op.drop_column("emo_conversations", "context_summary_until")
    op.drop_column("emo_conversations", "context_summary")
    op.drop_column("emo_user_settings", "ai_preferences")

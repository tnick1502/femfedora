"""perinatal context and pause reason"""

import sqlalchemy as sa

import emo_core.db.base
from alembic import op

revision = "f3a91c5d2b74"
down_revision = "d51c87126a40"
branch_labels = None
depends_on = None


def upgrade():
    """Добавляет исход беременности, ПДР, дату фиксации срока и причину паузы; заполняет старые записи."""
    op.add_column(
        "emo_user_settings", sa.Column("proactive_pause_reason", sa.String(24), nullable=True)
    )
    op.add_column(
        "emo_user_settings",
        sa.Column(
            "gestational_week_recorded_at",
            emo_core.db.base.UTCDateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column("emo_user_settings", sa.Column("estimated_due_date", sa.Date(), nullable=True))
    op.add_column("emo_user_settings", sa.Column("pregnancy_outcome", sa.String(16), nullable=True))
    # Ранее сохранённый срок привязывается к последнему изменению настроек: точнее данных нет.
    op.execute(
        "UPDATE emo_user_settings SET gestational_week_recorded_at = updated_at "
        "WHERE gestational_week IS NOT NULL"
    )
    # Существующие паузы считаются автоматическими, как и до миграции (снимаются при ответе).
    op.execute(
        "UPDATE emo_user_settings SET proactive_pause_reason = 'unanswered' "
        "WHERE proactive_paused_until IS NOT NULL"
    )


def downgrade():
    """Удаляет причину паузы и дополнительные поля перинатального контекста."""
    op.drop_column("emo_user_settings", "pregnancy_outcome")
    op.drop_column("emo_user_settings", "estimated_due_date")
    op.drop_column("emo_user_settings", "gestational_week_recorded_at")
    op.drop_column("emo_user_settings", "proactive_pause_reason")

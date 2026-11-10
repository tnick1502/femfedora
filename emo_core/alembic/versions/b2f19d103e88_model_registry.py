"""Global model registry, encrypted credentials and configuration audit."""

from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "b2f19d103e88"
down_revision = "e7e5ae1ff425"
branch_labels = None
depends_on = None


def upgrade():
    """Создаёт глобальные профили моделей, выбранные роли и журнал смены конфигурации."""
    op.create_table(
        "emo_model_profiles",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("model", sa.String(160), nullable=False),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("timeout_seconds", sa.Float(), nullable=False),
        sa.Column("credential_ref", sa.String(100)),
        sa.Column("encrypted_api_key", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_emo_model_profiles_created_at", "emo_model_profiles", ["created_at"])
    op.create_table(
        "emo_model_runtime",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("ai_mode", sa.String(16)),
        sa.Column("llm_profile_id", sa.Uuid(), sa.ForeignKey("emo_model_profiles.id")),
        sa.Column("jev_profile_id", sa.Uuid(), sa.ForeignKey("emo_model_profiles.id")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.bulk_insert(
        sa.table(
            "emo_model_runtime",
            sa.column("id", sa.Integer()),
            sa.column("revision", sa.Integer()),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        ),
        [{"id": 1, "revision": 0, "updated_at": datetime.now(UTC)}],
    )
    op.create_table(
        "emo_model_config_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("profile_id", sa.Uuid()),
        sa.Column(
            "metadata", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_emo_model_config_events_revision", "emo_model_config_events", ["revision"])
    op.create_index(
        "ix_emo_model_config_events_created_at", "emo_model_config_events", ["created_at"]
    )


def downgrade():
    """Удаляет журнал, текущий выбор и профили моделей вместе с сохранёнными ключами."""
    op.drop_table("emo_model_config_events")
    op.drop_table("emo_model_runtime")
    op.drop_table("emo_model_profiles")

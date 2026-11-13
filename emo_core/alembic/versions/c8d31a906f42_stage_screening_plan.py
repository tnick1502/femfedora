"""Stage-aware core scales, retire GAD-2 without deleting historical attempts."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "c8d31a906f42"
down_revision = "a7c2e94d1f36"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "emo_user_settings", sa.Column("reproductive_stage", sa.String(24), nullable=True)
    )
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    registry = sa.table(
        "emo_questionnaires",
        sa.column("id", sa.Uuid()),
        sa.column("code", sa.String()),
        sa.column("version", sa.String()),
        sa.column("language", sa.String()),
        sa.column("enabled", sa.Boolean()),
        sa.column("definition", json_type),
        sa.column("scoring_rules", json_type),
        sa.column("license_reference", sa.Text()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    bind = op.get_bind()
    # Never rewrite an existing questionnaire or attempt snapshot.
    materials = json.loads(
        (Path(__file__).resolve().parents[1] / "data/questionnaires_20261113_v2.json").read_text()
    )
    bind.execute(registry.update().where(registry.c.code == "GAD2").values(enabled=False))
    bundled_versions = {
        "ru-russia-pfizer-2026-11-10-v1",
        "ru-russia-pfizer-2026-11-13-v2",
        "ru-roag2024-source-2026-11-10-v1",
        "ru-yakupova2023-source-2026-11-13-v1",
        "unconfigured",
    }
    for material in materials:
        custom_active = (
            bind.execute(
                sa.select(registry.c.id).where(
                    registry.c.code == material["code"],
                    registry.c.language == material["language"],
                    registry.c.enabled.is_(True),
                    registry.c.version.not_in(bundled_versions),
                )
            ).first()
            is not None
        )
        existing = bind.execute(
            sa.select(registry.c.id).where(
                registry.c.code == material["code"],
                registry.c.version == material["version"],
                registry.c.language == material["language"],
            )
        ).scalar_one_or_none()
        if not custom_active:
            bind.execute(
                registry.update()
                .where(
                    registry.c.code == material["code"],
                    registry.c.language == material["language"],
                )
                .values(enabled=False)
            )
        if existing:
            if not custom_active:
                bind.execute(
                    registry.update()
                    .where(registry.c.id == existing)
                    .values(enabled=material["enabled"])
                )
        else:
            op.bulk_insert(
                registry,
                [
                    {
                        **material,
                        "enabled": material["enabled"] and not custom_active,
                        "id": uuid4(),
                        "created_at": datetime.now(UTC),
                    }
                ],
            )
    # Scheduling semantics changed: future dates will be recalculated from real completions.
    op.execute(
        sa.text(
            "UPDATE emo_assessment_schedule SET next_due_at=NULL, trigger_reason='stage_plan_v2' WHERE questionnaire_code IN ('PHQ9','GAD7','EPDS')"
        )
    )


def downgrade():
    # Medical history is retained on rollback. Re-enabling GAD-2 is an operator decision.
    op.execute(
        sa.text(
            "UPDATE emo_questionnaires SET enabled=false WHERE code='CITYBITS' OR version='ru-russia-pfizer-2026-11-13-v2'"
        )
    )
    op.execute(
        sa.text(
            "UPDATE emo_questionnaires AS previous SET enabled=true "
            "WHERE previous.code='PHQ9' AND previous.version='ru-russia-pfizer-2026-11-10-v1' "
            "AND NOT EXISTS (SELECT 1 FROM emo_questionnaires AS active "
            "WHERE active.code=previous.code AND active.language=previous.language "
            "AND active.enabled=true AND active.id<>previous.id)"
        )
    )
    op.drop_column("emo_user_settings", "reproductive_stage")

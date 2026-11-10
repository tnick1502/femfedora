"""Source-verified Russian forms and durable questionnaire workflow."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "d51c87126a40"
down_revision = "b2f19d103e88"
branch_labels = None
depends_on = None


def upgrade():
    """Добавляет черновикам ревизии, идемпотентность, снимок редакции и итоговый отчёт.

    Загружает зафиксированные проверенные формы, отключает заглушки unconfigured
    и заполняет отсутствующие снимки старых попыток. Существующие попытки сохраняет.
    """
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "emo_test_attempts",
        sa.Column("auxiliary_answers", json_type, nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "emo_test_attempts", sa.Column("revision", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column("emo_test_attempts", sa.Column("client_request_id", sa.Uuid()))
    op.add_column("emo_test_attempts", sa.Column("questionnaire_snapshot", json_type))
    op.add_column("emo_test_attempts", sa.Column("result", json_type))
    op.create_unique_constraint(
        "uq_attempt_user_request", "emo_test_attempts", ["user_id", "client_request_id"]
    )
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
    # This frozen migration data must never be edited when publishing a future version.
    path = Path(__file__).resolve().parents[1] / "data/questionnaires_20261110_v1.json"
    materials = json.loads(path.read_text())
    for data in materials:
        existing = (
            op.get_bind()
            .execute(
                sa.select(registry.c.id).where(
                    registry.c.code == data["code"],
                    registry.c.version == data["version"],
                    registry.c.language == data["language"],
                )
            )
            .scalar_one_or_none()
        )
        if existing:
            op.get_bind().execute(
                registry.update().where(registry.c.id == existing).values(enabled=True)
            )
        else:
            op.bulk_insert(registry, [{**data, "id": uuid4(), "created_at": datetime.now(UTC)}])
    # Keep existing registry entries and all attempts. Only obsolete placeholders are retired.
    op.execute(registry.update().where(registry.c.version == "unconfigured").values(enabled=False))
    attempts = sa.table(
        "emo_test_attempts",
        sa.column("questionnaire_id", sa.Uuid()),
        sa.column("questionnaire_snapshot", json_type),
    )
    bind = op.get_bind()
    for row in bind.execute(sa.select(registry)).mappings():
        material = {
            key: row[key]
            for key in (
                "code",
                "version",
                "language",
                "definition",
                "scoring_rules",
                "license_reference",
            )
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        snapshot = {
            **material,
            "id": str(row["id"]),
            "enabled": row["enabled"],
            "definition_hash": digest,
        }
        bind.execute(
            attempts.update()
            .where(
                attempts.c.questionnaire_id == row["id"],
                attempts.c.questionnaire_snapshot.is_(None),
            )
            .values(questionnaire_snapshot=snapshot)
        )


def downgrade():
    # History references remain intact; rollback retires rather than deletes these forms.
    """Отключает внесённые редакции и удаляет поля workflow, сохраняя записи шкал и попыток."""
    op.execute(
        sa.text(
            "UPDATE emo_questionnaires SET enabled=false "
            "WHERE version IN ('ru-roag2024-source-2026-11-10-v1', "
            "'ru-russia-pfizer-2026-11-10-v1')"
        )
    )
    op.drop_constraint("uq_attempt_user_request", "emo_test_attempts", type_="unique")
    for column in (
        "result",
        "questionnaire_snapshot",
        "client_request_id",
        "revision",
        "auxiliary_answers",
    ):
        op.drop_column("emo_test_attempts", column)

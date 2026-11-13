"""Disable PASS-R; history remains readable."""

import sqlalchemy as sa

from alembic import op

revision = "d92c45e17a60"
down_revision = "c8d31a906f42"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(sa.text("UPDATE emo_questionnaires SET enabled=false WHERE code='PASSR'"))
    op.execute(
        sa.text(
            "UPDATE emo_proactive_jobs SET status='cancelled', last_error='scale_inactive' WHERE status='pending' AND job_type='questionnaire_due' AND metadata->>'code'='PASSR'"
        )
    )


def downgrade():
    # Never silently reactivate a clinical material on rollback.
    pass

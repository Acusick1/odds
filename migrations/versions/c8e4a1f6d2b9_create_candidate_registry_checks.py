"""create candidate_registry_checks

Revision ID: c8e4a1f6d2b9
Revises: b5d2f8a3c6e1
Create Date: 2026-11-20 18:00:00.000000

One row per league with what its latest run-candidates run found wrong with the candidate
registry, which the health check reads.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c8e4a1f6d2b9"
down_revision = "b5d2f8a3c6e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidate_registry_checks",
        sa.Column("sport_key", sa.Text(), primary_key=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registry_error", sa.Text(), nullable=True),
        sa.Column(
            "invalid_candidates",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_table("candidate_registry_checks")

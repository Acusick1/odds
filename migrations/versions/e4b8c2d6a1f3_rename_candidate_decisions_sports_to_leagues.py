"""rename candidate_decisions.sports to leagues

Revision ID: e4b8c2d6a1f3
Revises: d5a3f7c19e24
Create Date: 2026-11-19 12:00:00.000000

Renames the column that holds the league keys a candidate version listed when it decided, to
match the candidate's ``leagues`` field. Nothing is dropped or copied; the downgrade renames it
back.
"""

from alembic import op

revision = "e4b8c2d6a1f3"
down_revision = "d5a3f7c19e24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("candidate_decisions", "sports", new_column_name="leagues")


def downgrade() -> None:
    op.alter_column("candidate_decisions", "leagues", new_column_name="sports")

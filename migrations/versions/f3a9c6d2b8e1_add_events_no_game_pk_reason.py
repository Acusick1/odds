"""add events.no_game_pk_reason, set by odds events mark-unassignable

Revision ID: f3a9c6d2b8e1
Revises: c9d4e1a7b3f2
Create Date: 2026-10-25 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "f3a9c6d2b8e1"
down_revision = "c9d4e1a7b3f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("events", sa.Column("no_game_pk_reason", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_events_game_pk_or_no_game_pk_reason",
        "events",
        "mlb_game_pk IS NULL OR no_game_pk_reason IS NULL",
    )


def downgrade() -> None:
    op.drop_constraint("ck_events_game_pk_or_no_game_pk_reason", "events", type_="check")
    op.drop_column("events", "no_game_pk_reason")

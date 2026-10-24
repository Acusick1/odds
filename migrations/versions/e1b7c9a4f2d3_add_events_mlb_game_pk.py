"""add events.mlb_game_pk with a partial unique index

Revision ID: e1b7c9a4f2d3
Revises: a5c8d2e91f37
Create Date: 2026-10-24 21:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "e1b7c9a4f2d3"
down_revision = "a5c8d2e91f37"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("events", sa.Column("mlb_game_pk", sa.BigInteger(), nullable=True))
    op.create_index(
        "uq_events_mlb_game_pk",
        "events",
        ["mlb_game_pk"],
        unique=True,
        postgresql_where=sa.text("mlb_game_pk IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_events_mlb_game_pk", table_name="events")
    op.drop_column("events", "mlb_game_pk")

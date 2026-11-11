"""record the kick-off an event was postponed from

Revision ID: e5b7c9d1f3a2
Revises: c4a8e1f6d2b9
Create Date: 2026-11-10 23:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "e5b7c9d1f3a2"
down_revision = "c4a8e1f6d2b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.create_table(
        "event_postponements",
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("postponed_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
    )


def downgrade() -> None:
    op.drop_table("event_postponements")

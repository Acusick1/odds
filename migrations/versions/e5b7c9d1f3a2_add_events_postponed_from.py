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
    op.add_column("events", sa.Column("postponed_from", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("events", "postponed_from")

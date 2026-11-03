"""add the sharp market's reading at placement to paper_trades

Revision ID: c4a8e1f6d2b9
Revises: b7e2d9a4c315
Create Date: 2026-11-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "c4a8e1f6d2b9"
down_revision = "b7e2d9a4c315"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("paper_trades", sa.Column("sharp_probability", sa.Float(), nullable=True))
    op.add_column("paper_trades", sa.Column("sharp_bookmaker", sa.String(), nullable=True))
    op.add_column(
        "paper_trades",
        sa.Column("sharp_snapshot_time", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("paper_trades", sa.Column("sharp_spread", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_trades", "sharp_spread")
    op.drop_column("paper_trades", "sharp_snapshot_time")
    op.drop_column("paper_trades", "sharp_bookmaker")
    op.drop_column("paper_trades", "sharp_probability")

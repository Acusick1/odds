"""one polymarket price snapshot per market and time, drop the unwritten book columns

Revision ID: a5c8d2e91f37
Revises: d4f1a8c6e2b9
Create Date: 2026-10-24 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "a5c8d2e91f37"
down_revision = "d4f1a8c6e2b9"
branch_labels = None
depends_on = None

TABLE = "polymarket_price_snapshots"


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.execute(
        sa.text(
            f"DELETE FROM {TABLE} a USING {TABLE} b"
            " WHERE a.polymarket_market_id = b.polymarket_market_id"
            " AND a.snapshot_time = b.snapshot_time AND a.id > b.id"
        )
    )
    op.drop_index("ix_pm_price_market_time", table_name=TABLE)
    op.create_unique_constraint(
        "uq_pm_price_market_time", TABLE, ["polymarket_market_id", "snapshot_time"]
    )
    for column in ("best_bid", "best_ask", "spread", "midpoint"):
        op.drop_column(TABLE, column)


def downgrade() -> None:
    for column in ("midpoint", "spread", "best_ask", "best_bid"):
        op.add_column(TABLE, sa.Column(column, sa.Float(), nullable=True))
    op.drop_constraint("uq_pm_price_market_time", TABLE, type_="unique")
    op.create_index(
        "ix_pm_price_market_time", TABLE, ["polymarket_market_id", "snapshot_time"], unique=False
    )

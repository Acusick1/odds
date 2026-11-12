"""record the planned Odds API historical snapshots the provider does not have

Revision ID: d2f8a4c6e1b9
Revises: b7d3f1a8c5e2
Create Date: 2026-11-12 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "d2f8a4c6e1b9"
down_revision = "b7d3f1a8c5e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.create_table(
        "odds_api_snapshot_gaps",
        sa.Column("sport_key", sa.Text(), nullable=False),
        sa.Column("market", sa.Text(), nullable=False),
        sa.Column("planned_snapshot", sa.DateTime(timezone=True), nullable=False),
        sa.Column("returned_snapshot", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_snapshot", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("sport_key", "market", "planned_snapshot"),
        sa.CheckConstraint(
            "market IN ('1x2', 'h2h', 'spreads', 'totals')",
            name="ck_odds_api_snapshot_gaps_market",
        ),
    )


def downgrade() -> None:
    op.drop_table("odds_api_snapshot_gaps")
